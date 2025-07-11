import torch

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 32]", primals_2: "f32[64]", primals_3: "Sym(s77)", primals_4: "f32[s77, 32]"):
         # File: <ipython-input-5-ade5b0d49fcc>:9 in forward, code: x = self.fc1(x)
        permute: "f32[32, 64]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f32[s77, 64]" = torch.ops.aten.addmm.default(primals_2, primals_4, permute);  primals_2 = permute = None
        
         # File: <ipython-input-5-ade5b0d49fcc>:10 in forward, code: x = torch.nn.functional.gelu(x)
        gelu: "f32[s77, 64]" = torch.ops.aten.gelu.default(addmm)
        return (gelu, primals_4, addmm, primals_3)
        
class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "Sym(s77)", primals_4: "f32[s77, 32]", addmm: "f32[s77, 64]", tangents_1: "f32[s77, 64]"):
        # No stacktrace found for following nodes
        fused_1 = self.fused_1(addmm)
        
         # File: <ipython-input-5-ade5b0d49fcc>:10 in forward, code: x = torch.nn.functional.gelu(x)
        getitem: "f32[s77, 64]" = fused_1[0]
        getitem_1: "f32[s77, 64]" = fused_1[1];  fused_1 = None
        erf: "f32[s77, 64]" = torch.ops.aten.erf.default(getitem);  getitem = None
        exp: "f32[s77, 64]" = torch.ops.aten.exp.default(getitem_1);  getitem_1 = None
        fused_0: "f32[s77, 64]" = self.fused_0(exp, erf, addmm, tangents_1);  exp = erf = addmm = tangents_1 = None
        
         # File: <ipython-input-5-ade5b0d49fcc>:9 in forward, code: x = self.fc1(x)
        permute_1: "f32[64, s77]" = torch.ops.aten.permute.default(fused_0, [1, 0])
        sum_1: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(fused_0, [0], True);  fused_0 = None
        mm: "f32[64, 32]" = torch.ops.aten.mm.default(permute_1, primals_4);  permute_1 = primals_4 = None
        view: "f32[64]" = torch.ops.aten.view.default(sum_1, [64]);  sum_1 = None
        permute_2: "f32[32, 64]" = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
        permute_3: "f32[64, 32]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return (permute_3, view, None, None)
        
    class fused_0(torch.nn.Module):
        def forward(self, exp: "f32[s77, 64]", erf: "f32[s77, 64]", addmm: "f32[s77, 64]", tangents_1: "f32[s77, 64]"):
             # File: <ipython-input-5-ade5b0d49fcc>:10 in forward, code: x = torch.nn.functional.gelu(x)
            mul_8: "f32[s77, 64]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
            add_6: "f32[s77, 64]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
            mul_9: "f32[s77, 64]" = torch.ops.aten.mul.Tensor(addmm, mul_8);  addmm = mul_8 = None
            mul_5: "f32[s77, 64]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
            add_7: "f32[s77, 64]" = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None
            mul_10: "f32[s77, 64]" = torch.ops.aten.mul.Tensor(tangents_1, add_7);  tangents_1 = add_7 = None
            return mul_10
            
    class fused_1(torch.nn.Module):
        def forward(self, addmm: "f32[s77, 64]"):
             # File: <ipython-input-5-ade5b0d49fcc>:10 in forward, code: x = torch.nn.functional.gelu(x)
            mul_6: "f32[s77, 64]" = torch.ops.aten.mul.Tensor(addmm, addmm)
            mul_4: "f32[s77, 64]" = torch.ops.aten.mul.Tensor(addmm, 0.7071067811865476);  addmm = None
            mul_7: "f32[s77, 64]" = torch.ops.aten.mul.Tensor(mul_6, -0.5);  mul_6 = None
            return (mul_4, mul_7)

a = GraphModule()
print()