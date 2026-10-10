# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
from vllm import LLM, SamplingParams
from vllm.parser.parser_manager import ParserManager

# Model checkpoints mapped to their registered vLLM tool parser name
TOOL_CALLING_MODELS = [
    ("ibm-granite/granite-3.3-8b-instruct", "granite"),
    ("ibm-granite/granite-4.1-8b", "hermes"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3_json"),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistral"),
    ("google/gemma-4-26B-A4B-it", "gemma4"),
]


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model,parser_name", TOOL_CALLING_MODELS)
def test_decoder_tool_calling(
    model: str,
    parser_name: str,
):
    """Verify tool calling works for decoder models using ToolParserManager."""
    # Initialize the LLM with the specified model.
    # We do not specify local chat_template path, vLLM will automatically
    # load the default built-in chat template from the model tokenizer.
    llm = LLM(
        model=model,
        max_model_len=512,
        max_num_seqs=1,
        tensor_parallel_size=1,
    )

    # Prepare user query and tool definition
    messages = [{"role": "user", "content": "List all files in the /tmp directory."}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Directory path"}},
                    "required": ["path"],
                },
            },
        }
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=400,
    )

    # Generate response
    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        tools=tools,
        use_tqdm=False,
    )

    generated_text = outputs[0].outputs[0].text
    print(f"Generated text: {generated_text}")

    # Dynamically get and initialize parser via ParserManager
    tokenizer = llm.get_tokenizer()
    parser_cls = ParserManager.get_tool_parser(parser_name, enable_auto_tools=True)
    parser = parser_cls(tokenizer=tokenizer)

    # Construct a dummy class for request since extract_tool_calls
    # takes a request object but does not access any of its properties.
    class DummyRequest:
        pass

    extracted = parser.extract_tool_calls(generated_text, DummyRequest())

    # Assert that tool calls were detected and successfully extracted
    assert extracted.tools_called, (
        f"Expected tool calls to be extracted, but parser returned: {extracted}"
    )
    assert hasattr(extracted, "tool_calls"), (
        "Extracted object does not have a 'tool_calls' attribute"
    )
    assert extracted.tool_calls is not None, "'tool_calls' attribute is None"
    assert len(extracted.tool_calls) > 0, "'tool_calls' list is empty"

    tool_call = extracted.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "list_files"

    # Parse and verify function arguments
    arguments = json.loads(tool_call.function.arguments)
    assert arguments.get("path") == "/tmp"
