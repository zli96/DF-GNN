import torch

def pattern_matching(graph):
    pattern = """ 
        graph(%x,%1,%2,%3): 
            %0:Tensor=aten::index_select(%x,%3,%1) 
            %out:Tensor=aten::mul(%0,%2)
            return(%out) 
    """ 
    
    replacement = """
        graph(%x,%1:Tensor,%2,%3):
            %12 : int = prim::Constant[value=1]()
            %49 : int = prim::Constant[value=2]() 
            %71 : int = prim::Constant[value=0]()
            %56 : bool = prim::Constant[value=0]()
            %48 : int = aten::dim(%x) 
            %50 : bool = aten::eq(%48, %49) 
            %57 : bool = prim::If(%50) 
                block0():
                    %53 : int = aten::dim(%1) 
                    %54 : bool = aten::eq(%53, %12) 
                    -> (%54)
                block1():
                    -> (%56)
            %63 : bool = prim::If(%57) 
                block0():
                    %59 : int = aten::dim(%2) 
                    %60 : bool = aten::eq(%59, %49) 
                    -> (%60)
                block1():
                    -> (%56)
            %x_kj : Tensor = prim::If(%63) 
                block0():
                    %x_kj.21 : Tensor = index_mul::index_mul_float(%x, %2, %1)
                    -> (%x_kj.21)
                block1():
                    %73 : Tensor = aten::index_select(%x, %71, %1) 
                    %x_kj.27 : Tensor = aten::mul(%73, %2) 
                    -> (%x_kj.27)
            return(%x_kj) 
    """

    torch._C._jit_pass_custom_pattern_based_rewrite_graph(pattern, replacement,graph) 

