# from transformers import LlamaConfig, LlamaForCausalLM as LlamaForCausalLM_Original

# #测试是否和官方模型的计算输出一样
# config = "{'vocab_size': 128256, 'max_position_embeddings': 8192, 'hidden_size': 1024, 'intermediate_size': 14336, 'num_hidden_layers': 4, 'num_attention_heads': 32, 'num_key_value_heads': 8, 'hidden_act': 'silu', 'initializer_range': 0.02, 'rms_norm_eps': 1e-05, 'pretraining_tp': 1, 'use_cache': False, 'rope_theta': 500000.0, 'rope_scaling': None, 'attention_bias': False, 'attention_dropout': 0.0, 'return_dict': True, 'output_hidden_states': False, 'output_attentions': False, 'torchscript': False, 'torch_dtype': 'bfloat16', 'use_bfloat16': False, 'tf_legacy_loss': False, 'pruned_heads': {}, 'tie_word_embeddings': False, 'chunk_size_feed_forward': 0, 'is_encoder_decoder': False, 'is_decoder': False, 'cross_attention_hidden_size': None, 'add_cross_attention': False, 'tie_encoder_decoder': False, 'max_length': 20, 'min_length': 0, 'do_sample': False, 'early_stopping': False, 'num_beams': 1, 'num_beam_groups': 1, 'diversity_penalty': 0.0, 'temperature': 1.0, 'top_k': 50, 'top_p': 1.0, 'typical_p': 1.0, 'repetition_penalty': 1.0, 'length_penalty': 1.0, 'no_repeat_ngram_size': 0, 'encoder_no_repeat_ngram_size': 0, 'bad_words_ids': None, 'num_return_sequences': 1, 'output_scores': False, 'return_dict_in_generate': False, 'forced_bos_token_id': None, 'forced_eos_token_id': None, 'remove_invalid_values': False, 'exponential_decay_length_penalty': None, 'suppress_tokens': None, 'begin_suppress_tokens': None, 'architectures': ['LlamaForCausalLM'], 'finetuning_task': None, 'id2label': {0: 'LABEL_0', 1: 'LABEL_1'}, 'label2id': {'LABEL_0': 0, 'LABEL_1': 1}, 'tokenizer_class': None, 'prefix': None, 'bos_token_id': 128000, 'pad_token_id': None, 'eos_token_id': 128001, 'sep_token_id': None, 'decoder_start_token_id': None, 'task_specific_params': None, 'problem_type': None, '_name_or_path': '', 'transformers_version': '4.38.2', 'model_type': 'llama'}"
# config = LlamaConfig.from_dict(eval(config))

# model_actor1 = LlamaForCausalLM_Original(config)
# model_actor2 = LlamaForCausalLM()

# model_actor2.load_state_dict(model_actor1.state_dict())

# input = {
#     'input_ids': torch.randint(100, 50000, [4, 125]),
#     'attention_mask': torch.ones(4, 125).long(),
#     'labels': torch.randint(100, 50000, [4, 125])
# }
# input['attention_mask'][:, 120:] = 0

# out = model_actor1(**input)
# loss, logits = model_actor2(**input)

# print(out.loss, out.logits.shape)
# print(loss, logits.shape)

# out.loss == loss, (out.logits == logits).all()