#include "../common/pytorch_cpp_helper.hpp"
#include "../common/pytorch_device_registry.hpp"
#include <torch/all.h>
#include <torch/python.h>
#include <torch/torch.h>

void index_mul_float_forward_impl(
    Tensor &out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    DISPATCH_DEVICE_IMPL(index_mul_float_forward_impl, out, in1, in2, idx1);
}

void index_mul_float_backward_impl(
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    DISPATCH_DEVICE_IMPL(index_mul_float_backward_impl, grad_in1, grad_in2, grad_out, in1, in2, idx1);
}

void index_mul_float_backward_backward_impl(
    Tensor &grad_grad_out,
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &grad_grad_in1,
    const Tensor &grad_grad_in2,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    DISPATCH_DEVICE_IMPL(index_mul_float_backward_backward_impl, grad_grad_out, grad_in1, grad_in2, grad_out,
                         grad_grad_in1, grad_grad_in2, in1, in2, idx1);
}

void index_mul_float_forward(
    Tensor &out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    index_mul_float_forward_impl(out, in1, in2, idx1);
}

void index_mul_float_backward(
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    index_mul_float_backward_impl(grad_in1, grad_in2, grad_out, in1, in2, idx1);
}

void index_mul_float_backward_backward(
    Tensor &grad_grad_out,
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &grad_grad_in1,
    const Tensor &grad_grad_in2,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    index_mul_float_backward_backward_impl(grad_grad_out, grad_in1, grad_in2, grad_out, grad_grad_in1, grad_grad_in2, in1, in2, idx1);
}

class IndexMulBackward : public torch::autograd::Function<IndexMulBackward>
{
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor &in1,
        torch::Tensor &in2,
        torch::Tensor &idx1,
        torch::Tensor &grad_out)
    {
        // forward calculation
        auto grad_in1 = torch::zeros_like(in1);
        auto grad_in2 = torch::empty_like(in2);
        index_mul_float_backward_impl(grad_in1, grad_in2, grad_out, in1, in2, idx1);
        ctx->save_for_backward({in1, in2, idx1, grad_out});
        return {grad_in1, grad_in2};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list &grad_grad_in)
    {
        auto saved = ctx->get_saved_variables();
        auto in1 = saved[0];
        auto in2 = saved[1];
        auto idx1 = saved[2];
        auto grad_out = saved[3];
        auto grad_grad_out = torch::empty_like(grad_out);
        auto grad_in1 = torch::zeros_like(in1);
        auto grad_in2 = torch::empty_like(in2);
        auto grad_grad_in1 = grad_grad_in[0];
        auto grad_grad_in2 = grad_grad_in[1];
        index_mul_float_backward_backward_impl(grad_grad_out, grad_in1, grad_in2, grad_out, grad_grad_in1, grad_grad_in2, in1, in2, idx1);
        return {grad_in1, grad_in2, torch::Tensor(), grad_grad_out};
    }
};

torch::autograd::tensor_list IndexMulBackward_op(
    torch::Tensor &in1,
    torch::Tensor &in2,
    torch::Tensor &idx1,
    torch::Tensor &grad_out)
{
    return IndexMulBackward::apply(in1, in2, idx1, grad_out);
}

class IndexMul : public torch::autograd::Function<IndexMul>
{
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor &in1,
        const torch::Tensor &in2,
        const torch::Tensor &idx1)
    {
        // forward calculation
        torch::Tensor out = torch::empty_like(in2);
        index_mul_float_forward_impl(out, in1, in2, idx1);
        ctx->save_for_backward({in1, in2, idx1});
        printf("run index+mul successful\n");
        return out;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list &grad_out)
    {
        auto saved = ctx->get_saved_variables();
        auto in1 = saved[0];
        auto in2 = saved[1];
        auto idx1 = saved[2];
        auto grad_in1 = torch::zeros_like(in1);
        auto grad_in2 = torch::empty_like(in2);
        torch::autograd::tensor_list grad = IndexMulBackward_op(in1, in2, idx1, grad_out[0]);
        grad_in1 = grad[0];
        grad_in2 = grad[1];
        return {grad_in1, grad_in2, torch::Tensor()};
    }
};

torch::Tensor IndexMul_op(
    const torch::Tensor &in1,
    const torch::Tensor &in2,
    const torch::Tensor &idx1)
{
    return IndexMul::apply(in1, in2, idx1);
}
