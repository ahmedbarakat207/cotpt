import importlib.util
import os


spec = importlib.util.spec_from_file_location("cotpt_main", os.path.abspath("main.py"))
cotpt_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cotpt_main)
build_arg_parser = cotpt_main.build_arg_parser
format_chat_prompt = cotpt_main.format_chat_prompt


def test_build_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.num_hidden_tokens > 0
    assert args.show_hidden_thoughts is True
    assert args.system_prompt != ""


def test_format_chat_prompt_fallback(mock_tokenizer):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    prompt = format_chat_prompt(mock_tokenizer, messages, fallback_system="System test")
    assert "System: System test" in prompt
    assert "User: Hello" in prompt
    assert "Assistant: Hi there" in prompt
    assert "User: What is 2+2?" in prompt
    assert prompt.endswith("Assistant: ")
