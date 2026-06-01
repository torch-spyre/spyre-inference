<h1 align="center">
Spyre Inference
</h1>

<p align="center">
| <a href="https://torch-spyre.github.io/spyre-inference/"><b>Documentation</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/"><b>Users Forum</b></a> | <a href="https://slack.vllm.ai"><b>#sig-spyre</b></a> |
</p>

---

**IBM Spyre** is the first production-grade Artificial Intelligence Unit (AIU) accelerator born out of the IBM Research AIU family, and is part of a long-term strategy of developing novel architectures and full-stack technology solutions for the emerging space of generative AI. Spyre builds on the foundation of IBM's internal AIU research and delivers a scalable, efficient architecture for accelerating AI in enterprise environments.

`spyre-inference` is a vLLM platform plugin that enables seamless integration of IBM Spyre accelerators with vLLM via the [`torch-spyre`](https://github.com/torch-spyre/torch-spyre) PyTorch backend. It is the next evolution of [`sendnn-inference`](https://github.com/torch-spyre/sendnn-inference), leveraging PyTorch's native Inductor compiler backend through vLLM's plugin architecture.

For more information, check out the following:

- 📚 [Meet the IBM Artificial Intelligence Unit](https://research.ibm.com/blog/ibm-artificial-intelligence-unit-aiu)
- 📽️ [AI Accelerators: Transforming Scalability & Model Efficiency](https://www.youtube.com/watch?v=KX0qBM-ByAg)
- 🚀 [Spyre Accelerator for IBM Z](https://research.ibm.com/blog/spyre-for-z)
- 🚀 [Spyre Accelerator for IBM POWER](https://newsroom.ibm.com/2025-07-08-ibm-power11-raises-the-bar-for-enterprise-it)

## Getting Started

Visit our [documentation](https://torch-spyre.github.io/spyre-inference/):

- [Installation](https://torch-spyre.github.io/spyre-inference/latest/getting_started/installation.html)
- [Examples](https://torch-spyre.github.io/spyre-inference/latest/examples/offline_inference/)
- [Contributing Guide](https://torch-spyre.github.io/spyre-inference/latest/contributing/)

## Contributing

We welcome and value any contributions and collaborations. Please check out [Contributing to Spyre Inference](https://torch-spyre.github.io/spyre-inference/latest/contributing/) for how to get involved.

## Contact

You can reach out for discussion or support in the `#sig-spyre` channel in the [vLLM Slack](https://inviter.co/vllm-slack) workspace or by [opening an issue](https://github.com/torch-spyre/spyre-inference/issues).

## License

[Apache-2.0](LICENSE)
