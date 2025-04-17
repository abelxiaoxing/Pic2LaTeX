from pathlib import Path
import onnxruntime
import numpy as np
from transformers import (
    RobertaTokenizerFast
)

class Pic2LaTeX():
    REPO_NAME = './checkpoints'
    def __init__(self):
        self.model = ONNXModel.from_pretrained(self.REPO_NAME, onnx_provider)
        self.tokenizer = self.get_tokenizer()
    
    @classmethod
    def from_pretrained(cls, onnx_provider=None):
        return ONNXModel(cls.REPO_NAME, onnx_provider)

    @classmethod
    def get_tokenizer(cls) -> RobertaTokenizerFast:
        return RobertaTokenizerFast.from_pretrained(cls.REPO_NAME)


class ONNXModel:
    """使用原生onnxruntime替代optimum.onnxruntime.ORTModelForVision2Seq"""
 
  
    def __init__(self, model_path, onnx_provider=None):
        self.model_path = Path(model_path)
        
        # 设置ONNX运行时提供程序
        self.providers = ['CPUExecutionProvider']
        if onnx_provider == 'cuda' and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
        
        # 加载encoder和decoder模型
        encoder_path = self.model_path / 'encoder_model.onnx'
        decoder_path = self.model_path / 'decoder_model.onnx'
        
        self.encoder = onnxruntime.InferenceSession(str(encoder_path), providers=self.providers)
        self.decoder = onnxruntime.InferenceSession(str(decoder_path), providers=self.providers)
        
        # 获取输入输出名称
        self.encoder_input_name = self.encoder.get_inputs()[0].name
        self.encoder_output_name = self.encoder.get_outputs()[0].name
    
    def to(self, device):
        return self
    
    def generate(self, pixel_values, generation_config=None):
        
        batch_size = pixel_values.shape[0]
        
        # 运行encoder
        encoder_outputs = self.encoder.run(
            None, {self.encoder_input_name: pixel_values}
        )[0]
        
        # 获取生成配置参数
        max_length = generation_config.max_new_tokens
        num_beams = generation_config.num_beams
        eos_token_id = generation_config.eos_token_id
        bos_token_id = generation_config.bos_token_id
        pad_token_id = generation_config.pad_token_id
        
        # 初始化decoder输入
        if num_beams > 1:
            # Beam Search解码
            return self._generate_beam_search(
                encoder_outputs, 
                batch_size,
                max_length, 
                num_beams, 
                bos_token_id, 
                eos_token_id, 
                pad_token_id
            )
        else:
            # 贪婪解码
            return self._generate_greedy(
                encoder_outputs, 
                batch_size,
                max_length, 
                bos_token_id, 
                eos_token_id
            )
    
    def _generate_greedy(self, encoder_outputs, batch_size, max_length, bos_token_id, eos_token_id):
        """贪婪解码实现"""
        # 初始化decoder输入
        input_ids = np.ones((batch_size, 1), dtype=np.int64) * bos_token_id
        
        for _ in range(max_length):
            # 运行decoder
            outputs = self.decoder.run(
                None, {
                    "input_ids": input_ids,
                    "encoder_hidden_states": encoder_outputs
                }
            )
            
            # 获取下一个token
            next_token_logits = outputs[0][:, -1, :]
            next_tokens = np.argmax(next_token_logits, axis=-1)
            
            # 添加预测的token
            next_tokens = next_tokens.reshape(-1, 1)
            input_ids = np.concatenate([input_ids, next_tokens], axis=-1)
            
            # 检查是否生成了EOS token
            if (next_tokens == eos_token_id).all():
                break
        
        return input_ids
    
    def _generate_beam_search(self, encoder_outputs, batch_size, max_length, num_beams, bos_token_id, eos_token_id, pad_token_id):
        """简化版Beam Search实现"""

        
        # 为每个样本创建beam_size个候选序列
        input_ids = np.ones((batch_size * num_beams, 1), dtype=np.int64) * bos_token_id
        
        # 复制encoder_outputs以匹配beam size
        encoder_outputs_expanded = np.repeat(encoder_outputs, num_beams, axis=0)
        
        # 跟踪每个beam的分数
        beam_scores = np.zeros((batch_size, num_beams), dtype=np.float32)
        
        # 跟踪已完成的序列
        done = [False for _ in range(batch_size)]
        
        for step in range(max_length):
            # 运行decoder
            outputs = self.decoder.run(
                None, {
                    "input_ids": input_ids,
                    "encoder_hidden_states": encoder_outputs_expanded
                }
            )
            
            # 获取logits
            next_token_logits = outputs[0][:, -1, :]
            
            # 将logits转换为对数概率
            # 手动实现log_softmax，因为numpy没有这个函数
            logits_max = np.max(next_token_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(next_token_logits - logits_max)
            exp_logits_sum = np.sum(exp_logits, axis=-1, keepdims=True)
            next_token_scores = (next_token_logits - logits_max) - np.log(exp_logits_sum)  # (batch_size * num_beams, vocab_size)
            
            next_token_scores = next_token_scores.reshape(batch_size, num_beams, -1)  # (batch_size, num_beams, vocab_size)
            vocab_size = next_token_scores.shape[-1]
            
            # 计算下一个候选序列的分数
            next_scores = beam_scores[:, :, np.newaxis] + next_token_scores  # (batch_size, num_beams, vocab_size)
            next_scores = next_scores.reshape(batch_size, -1)  # (batch_size, num_beams * vocab_size)
            
            # 获取每个样本的top-k分数和对应的token索引
            topk_scores, topk_indices = [], []
            for i in range(batch_size):
                if done[i]:
                    # 如果该样本已完成，保持不变
                    _scores = np.zeros(num_beams) - 1e9
                    _scores[0] = 0.0
                    _indices = np.ones(num_beams, dtype=np.int64) * pad_token_id
                else:
                    # 获取top-k
                    _scores = next_scores[i]
                    _indices = np.argsort(-_scores)[:num_beams]
                    _scores = _scores[_indices]
                
                topk_scores.append(_scores)
                topk_indices.append(_indices)
            
            topk_scores = np.array(topk_scores)  # (batch_size, num_beams)
            topk_indices = np.array(topk_indices)  # (batch_size, num_beams)
            
            # 将token索引转换为词汇表索引和beam索引
            topk_tokens = topk_indices % vocab_size  # (batch_size, num_beams)
            topk_beam_indices = topk_indices // vocab_size  # (batch_size, num_beams)
            
            # 更新beam分数
            beam_scores = topk_scores
            
            # 准备下一步的输入ID
            next_input_ids = []
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 如果该样本已完成，保持不变
                    next_input_ids.append(input_ids[batch_idx * num_beams:(batch_idx + 1) * num_beams])
                    continue
                
                # 检查是否有序列完成
                for beam_idx, token_id in enumerate(topk_tokens[batch_idx]):
                    if token_id == eos_token_id:
                        done[batch_idx] = True
                
                # 获取新的输入ID
                batch_beam_indices = batch_idx * num_beams + topk_beam_indices[batch_idx]
                batch_beam_tokens = topk_tokens[batch_idx].reshape(-1, 1)
                
                batch_input_ids = []
                for i, beam_idx in enumerate(batch_beam_indices):
                    # 获取当前beam的历史序列
                    prev_ids = input_ids[beam_idx]
                    # 添加新token
                    new_ids = np.concatenate([prev_ids, batch_beam_tokens[i]], axis=0)
                    batch_input_ids.append(new_ids)
                
                next_input_ids.append(np.stack(batch_input_ids))
            
            # 更新input_ids
            input_ids = np.vstack(next_input_ids)
            
            # 检查是否所有样本都完成
            if all(done):
                break
        
        # 返回每个样本的最佳序列（第一个beam）
        result = []
        for i in range(batch_size):
            result.append(input_ids[i * num_beams])
        
        return np.stack(result)
