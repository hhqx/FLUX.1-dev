import os
import importlib
import torch
from typing import Optional, Union
from diffusers.utils import logging,is_accelerate_available,is_torch_version
from diffusers.pipelines.pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    CONNECTED_PIPES_KEYS,
    CUSTOM_PIPELINE_FILE_NAME,
    LOADABLE_CLASSES,
    _fetch_class_library_tuple,
    _get_custom_components_and_folders,
    _get_custom_pipeline_class,
    _get_final_device_map,
    _get_ignore_patterns,
    _get_pipeline_class,
    _identify_model_variants,
    _maybe_raise_warning_for_inpainting,
    _resolve_custom_pipeline_and_cls,
    _unwrap_model,
    _update_init_kwargs_with_connected_pipeline,
    load_sub_model,
    maybe_raise_or_warn,
    variant_compatible_siblings,
    warn_deprecated_model_variant,
)
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT

logger = logging.get_logger(__name__)


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    @staticmethod
    def allreduce(*args, **kwargs):
        return FakeBarrier()

    @staticmethod
    def allgather(inputs, local_tensor, **kwargs):
        for input_ in inputs:
            input_[0].data = local_tensor[0].data
        return FakeBarrier()

    @staticmethod
    def barrier(*args, **kwargs):
        return FakeBarrier()

    def size(self):
        return self._size

    def rank(self):
        return self._rank

def initialize_torch_distributed(rank, world_size):
    if not torch.npu.is_available():
        raise NotImplementedError("NPU is not available, please check device and torch_npu library")

    from torch_npu._C._distributed_c10d import ProcessGroupHCCL

    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)

    backend = "hccl"
    options = ProcessGroupHCCL.Options()
    print("ProcessGroupHCCL has been Set")

    if world_size == 1:
        return FakeGroup(rank, world_size), device
    else:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
                pg_options=options,
            )
            print(f"rank {rank} init {torch.distributed.is_initialized()},init_process_group has benn activated")
        else:
            print("torch.distributed is already initialized.")
    
    return torch.distributed.group.WORLD, device

def get_local_rank():
    return int(os.environ["LOCAL_RANK"])

def get_world_size():
    return int(os.environ["WORLD_SIZE"])

def replace_tp_from_pretrain(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
    kwargs_copied = kwargs.copy()

    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    from_flax = kwargs.pop("from_flax", False)
    torch_dtype = kwargs.pop("torch_dtype", None)
    custom_pipeline = kwargs.pop("custom_pipeline", None)
    custom_revision = kwargs.pop("custom_revision", None)
    provider = kwargs.pop("provider", None)
    sess_options = kwargs.pop("sess_options", None)
    device_map = kwargs.pop("device_map", None)
    max_memory = kwargs.pop("max_memory", None)
    offload_folder = kwargs.pop("offload_folder", None)
    offload_state_dict = kwargs.pop("offload_state_dict", False)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
    variant = kwargs.pop("variant", None)
    use_safetensors = kwargs.pop("use_safetensors", None)
    use_onnx = kwargs.pop("use_onnx", None)
    load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)

    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
            " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
            " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
            " install accelerate\n```\n."
        )

    if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `low_cpu_mem_usage=False`."
        )

    if device_map is not None and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `device_map=None`."
        )

    if device_map is not None and not is_accelerate_available():
        raise NotImplementedError(
            "Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`."
        )

    if device_map is not None and not isinstance(device_map, str):
        raise ValueError("`device_map` must be a string.")

    if device_map is not None and device_map not in SUPPORTED_DEVICE_MAP:
        raise NotImplementedError(
            f"{device_map} not supported. Supported strategies are: {', '.join(SUPPORTED_DEVICE_MAP)}"
        )

    if device_map is not None and device_map in SUPPORTED_DEVICE_MAP:
        if is_accelerate_version("<", "0.28.0"):
            raise NotImplementedError("Device placement requires `accelerate` version `0.28.0` or later.")

    if low_cpu_mem_usage is False and device_map is not None:
        raise ValueError(
            f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
            " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
        )

    # 1. Download the checkpoints and configs
    # use snapshot download here to get it working from from_pretrained
    if not os.path.isdir(pretrained_model_name_or_path):
        if pretrained_model_name_or_path.count("/") > 1:
            raise ValueError(
                f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                " is neither a valid local path nor a valid repo id. Please check the parameter."
            )
        cached_folder = cls.download(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            from_flax=from_flax,
            use_safetensors=use_safetensors,
            use_onnx=use_onnx,
            custom_pipeline=custom_pipeline,
            custom_revision=custom_revision,
            variant=variant,
            load_connected_pipeline=load_connected_pipeline,
            **kwargs,
        )
    else:
        cached_folder = pretrained_model_name_or_path

    # The variant filenames can have the legacy sharding checkpoint format that we check and throw
    # a warning if detected.
    if variant is not None and _check_legacy_sharding_variant_format(folder=cached_folder, variant=variant):
        warn_msg = (
            f"Warning: The repository contains sharded checkpoints for variant '{variant}' maybe in a deprecated format. "
            "Please check your files carefully:\n\n"
            "- Correct format example: diffusion_pytorch_model.fp16-00003-of-00003.safetensors\n"
            "- Deprecated format example: diffusion_pytorch_model-00001-of-00002.fp16.safetensors\n\n"
            "If you find any files in the deprecated format:\n"
            "1. Remove all existing checkpoint files for this variant.\n"
            "2. Re-obtain the correct files by running `save_pretrained()`.\n\n"
            "This will ensure you're using the most up-to-date and compatible checkpoint format."
        )
        logger.warning(warn_msg)

    config_dict = cls.load_config(cached_folder)

    # pop out "_ignore_files" as it is only needed for download
    config_dict.pop("_ignore_files", None)

    # 2. Define which model components should load variants
    # We retrieve the information by matching whether variant model checkpoints exist in the subfolders.
    # Example: `diffusion_pytorch_model.safetensors` -> `diffusion_pytorch_model.fp16.safetensors`
    # with variant being `"fp16"`.
    model_variants = _identify_model_variants(folder=cached_folder, variant=variant, config=config_dict)
    if len(model_variants) == 0 and variant is not None:
        error_message = f"You are trying to load the model files of the `variant={variant}`, but no such modeling files are available."
        raise ValueError(error_message)

    # 3. Load the pipeline class, if using custom module then load it from the hub
    # if we load from explicit class, let's use it
    local_rank = int(os.environ["LOCAL_RANK"])
    transformer_path = "transformer_" + str(local_rank)
    if 'transformer' in config_dict:
        config_dict[transformer_path] = config_dict.pop("transformer")
    custom_pipeline, custom_class_name = _resolve_custom_pipeline_and_cls(
        folder=cached_folder, config=config_dict, custom_pipeline=custom_pipeline
    )
    pipeline_class = _get_pipeline_class(
        cls,
        config=config_dict,
        load_connected_pipeline=load_connected_pipeline,
        custom_pipeline=custom_pipeline,
        class_name=custom_class_name,
        cache_dir=cache_dir,
        revision=custom_revision,
    )

    if device_map is not None and pipeline_class._load_connected_pipes:
        raise NotImplementedError("`device_map` is not yet supported for connected pipelines.")

    # DEPRECATED: To be removed in 1.0.0
    # we are deprecating the `StableDiffusionInpaintPipelineLegacy` pipeline which gets loaded
    # when a user requests for a `StableDiffusionInpaintPipeline` with `diffusers` version being <= 0.5.1.
    _maybe_raise_warning_for_inpainting(
        pipeline_class=pipeline_class,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config=config_dict,
    )

    # 4. Define expected modules given pipeline signature
    # and define non-None initialized modules (=`init_kwargs`)

    # some modules can be passed directly to the init
    # in this case they are already instantiated in `kwargs`
    # extract them here
    expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
    if 'transformer' in expected_modules:
        expected_modules.remove('transformer')
        expected_modules.add(transformer_path)
    expected_types = pipeline_class._get_signature_types()
    passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
    passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
    init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

    # define init kwargs and make sure that optional component modules are filtered out
    init_kwargs = {
        k: init_dict.pop(k)
        for k in optional_kwargs
        if k in init_dict and k not in pipeline_class._optional_components
    }
    init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

    # remove `null` components
    def load_module(name, value):
        if value[0] is None:
            return False
        if name in passed_class_obj and passed_class_obj[name] is None:
            return False
        return True

    init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

    for key in init_dict.keys():
        if key not in passed_class_obj:
            continue
        if "scheduler" in key:
            continue

        class_obj = passed_class_obj[key]
        _expected_class_types = []
        for expected_type in expected_types[key]:
            if isinstance(expected_type, enum.EnumMeta):
                _expected_class_types.extend(expected_type.__members__.keys())
            else:
                _expected_class_types.append(expected_type.__name__)

        _is_valid_type = class_obj.__class__.__name__ in _expected_class_types
        if not _is_valid_type:
            logger.warning(
                f"Expected types for {key}: {_expected_class_types}, got {class_obj.__class__.__name__}."
            )

    # Special case: safety_checker must be loaded separately when using `from_flax`
    if from_flax and "safety_checker" in init_dict and "safety_checker" not in passed_class_obj:
        raise NotImplementedError(
            "The safety checker cannot be automatically loaded when loading weights `from_flax`."
            " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
            " separately if you need it."
        )

    # 5. Throw nice warnings / errors for fast accelerate loading
    if len(unused_kwargs) > 0:
        logger.warning(
            f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
        )

    # import it here to avoid circular import
    from diffusers import pipelines

    # 6. device map delegation
    final_device_map = None
    if device_map is not None:
        final_device_map = _get_final_device_map(
            device_map=device_map,
            pipeline_class=pipeline_class,
            passed_class_obj=passed_class_obj,
            init_dict=init_dict,
            library=library,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
            cached_folder=cached_folder,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )

    # 7. Load each module in the pipeline
    current_device_map = None
    for name, (library_name, class_name) in logging.tqdm(init_dict.items(), desc="Loading pipeline components..."):
        # 7.1 device_map shenanigans
        if final_device_map is not None and len(final_device_map) > 0:
            component_device = final_device_map.get(name, None)
            if component_device is not None:
                current_device_map = {"": component_device}
            else:
                current_device_map = None

        # 7.2 - now that JAX/Flax is an official framework of the library, we might load from Flax names
        class_name = class_name[4:] if class_name.startswith("Flax") else class_name

        # 7.3 Define all importable classes
        is_pipeline_module = hasattr(pipelines, library_name)
        importable_classes = ALL_IMPORTABLE_CLASSES
        loaded_sub_model = None

        # 7.4 Use passed sub model or load class_name from library_name
        if name in passed_class_obj:
            # if the model is in a pipeline module, then we load it from the pipeline
            # check that passed_class_obj has correct parent class
            maybe_raise_or_warn(
                library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
            )

            loaded_sub_model = passed_class_obj[name]
        else:
            # load sub model
            loaded_sub_model = load_sub_model(
                library_name=library_name,
                class_name=class_name,
                importable_classes=importable_classes,
                pipelines=pipelines,
                is_pipeline_module=is_pipeline_module,
                pipeline_class=pipeline_class,
                torch_dtype=torch_dtype,
                provider=provider,
                sess_options=sess_options,
                device_map=current_device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                model_variants=model_variants,
                name=name,
                from_flax=from_flax,
                variant=variant,
                low_cpu_mem_usage=low_cpu_mem_usage,
                cached_folder=cached_folder,
                use_safetensors=use_safetensors,
            )
            logger.info(
                f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
            )

        init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

    # 8. Handle connected pipelines.
    if pipeline_class._load_connected_pipes and os.path.isfile(os.path.join(cached_folder, "README.md")):
        init_kwargs = _update_init_kwargs_with_connected_pipeline(
            init_kwargs=init_kwargs,
            passed_pipe_kwargs=passed_pipe_kwargs,
            passed_class_objs=passed_class_obj,
            folder=cached_folder,
            **kwargs_copied,
        )

    # 9. Potentially add passed objects if expected
    missing_modules = set(expected_modules) - set(init_kwargs.keys())
    passed_modules = list(passed_class_obj.keys())
    optional_modules = pipeline_class._optional_components
    if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
        for module in missing_modules:
            init_kwargs[module] = passed_class_obj.get(module, None)
    elif len(missing_modules) > 0:
        passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
        raise ValueError(
            f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
        )

    # 10. Instantiate the pipeline
    if transformer_path in init_kwargs:
        init_kwargs['transformer'] = init_kwargs.pop(transformer_path)
    model = pipeline_class(**init_kwargs)

    # 11. Save where the model was instantiated from
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    if device_map is not None:
        setattr(model, "hf_device_map", final_device_map)
    return model

def replace_tp_extract_init_dict(cls, config_dict, **kwargs):
    # Skip keys that were not present in the original config, so default __init__ values were used
    used_defaults = config_dict.get("_use_default_values", [])
    config_dict = {k: v for k, v in config_dict.items() if k not in used_defaults and k != "_use_default_values"}

    # 0. Copy origin config dict
    original_dict = dict(config_dict.items())

    # 1. Retrieve expected config attributes from __init__ signature
    expected_keys = cls._get_init_keys(cls)
    local_rank = int(os.environ["LOCAL_RANK"])
    if 'transformer' in expected_keys:
        expected_keys.remove('transformer')
        transformer_path = 'transformer_' + str(local_rank)
        expected_keys.add(transformer_path)
    expected_keys.remove("self")
    # remove general kwargs if present in dict
    if "kwargs" in expected_keys:
        expected_keys.remove("kwargs")
    # remove flax internal keys
    if hasattr(cls, "_flax_internal_args"):
        for arg in cls._flax_internal_args:
            expected_keys.remove(arg)

    # 2. Remove attributes that cannot be expected from expected config attributes
    # remove keys to be ignored
    if len(cls.ignore_for_config) > 0:
        expected_keys = expected_keys - set(cls.ignore_for_config)

    # load diffusers library to import compatible and original scheduler
    diffusers_library = importlib.import_module(__name__.split(".")[0])

    if cls.has_compatibles:
        compatible_classes = [c for c in cls._get_compatibles() if not isinstance(c, DummyObject)]
    else:
        compatible_classes = []

    expected_keys_comp_cls = set()
    for c in compatible_classes:
        expected_keys_c = cls._get_init_keys(c)
        expected_keys_comp_cls = expected_keys_comp_cls.union(expected_keys_c)
    expected_keys_comp_cls = expected_keys_comp_cls - cls._get_init_keys(cls)
    config_dict = {k: v for k, v in config_dict.items() if k not in expected_keys_comp_cls}

    # remove attributes from orig class that cannot be expected
    orig_cls_name = config_dict.pop("_class_name", cls.__name__)
    if (
        isinstance(orig_cls_name, str)
        and orig_cls_name != cls.__name__
        and hasattr(diffusers_library, orig_cls_name)
    ):
        orig_cls = getattr(diffusers_library, orig_cls_name)
        unexpected_keys_from_orig = cls._get_init_keys(orig_cls) - expected_keys
        config_dict = {k: v for k, v in config_dict.items() if k not in unexpected_keys_from_orig}
    elif not isinstance(orig_cls_name, str) and not isinstance(orig_cls_name, (list, tuple)):
        raise ValueError(
            "Make sure that the `_class_name` is of type string or list of string (for custom pipelines)."
        )

    # remove private attributes
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}

    # remove quantization_config
    config_dict = {k: v for k, v in config_dict.items() if k != "quantization_config"}

    # 3. Create keyword arguments that will be passed to __init__ from expected keyword arguments
    init_dict = {}
    for key in expected_keys:
        # if config param is passed to kwarg and is present in config dict
        # it should overwrite existing config dict key
        if key in kwargs and key in config_dict:
            config_dict[key] = kwargs.pop(key)

        if key in kwargs:
            # overwrite key
            init_dict[key] = kwargs.pop(key)
        elif key in config_dict:
            # use value from config dict
            init_dict[key] = config_dict.pop(key)

    # 4. Give nice warning if unexpected values have been passed
    if len(config_dict) > 0:
        logger.warning(
            f"The config attributes {config_dict} were passed to {cls.__name__}, "
            "but are not expected and will be ignored. Please verify your "
            f"{cls.config_name} configuration file."
        )

    # 5. Give nice info if config attributes are initialized to default because they have not been passed
    passed_keys = set(init_dict.keys())
    if len(expected_keys - passed_keys) > 0:
        logger.info(
            f"{expected_keys - passed_keys} was not found in config. Values will be initialized to default values."
        )

    # 6. Define unused keyword arguments
    unused_kwargs = {**config_dict, **kwargs}

    # 7. Define "hidden" config parameters that were saved for compatible classes
    hidden_config_dict = {k: v for k, v in original_dict.items() if k not in init_dict}

    return init_dict, unused_kwargs, hidden_config_dict