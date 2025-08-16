"""
ONNX Wrapper только для политики - без value function
Исключает global_state который нужен только для value
"""

import torch
import torch.nn as nn

class PolicyOnlyONNXWrapper(torch.nn.Module):
    """Wrapper для экспорта только политики (без value function)"""
    
    def __init__(self, original_model, onnx_model, ne: int, na: int):
        super().__init__()
        self.onnx_model = onnx_model
        self.ne = ne
        self.na = na
        
        # Переносим веса из оригинальной модели
        self._transfer_weights(original_model, onnx_model)
        
    def _transfer_weights(self, original_model, onnx_model):
        """Переносит веса из оригинальной модели в ONNX-совместимую"""
        print("Transferring weights to ONNX model...")
        
        original_state = original_model.state_dict()
        onnx_state = onnx_model.state_dict()
        
        transferred = 0
        total_params = len(onnx_state)
        
        for onnx_name, onnx_param in onnx_state.items():
            transferred_this_param = False
            
            # 1. Прямое совпадение имен
            if onnx_name in original_state:
                orig_param = original_state[onnx_name]
                if orig_param.shape == onnx_param.shape:
                    onnx_param.data.copy_(orig_param.data.cpu())
                    transferred += 1
                    transferred_this_param = True
                else:
                    print(f"Shape mismatch for {onnx_name}: {orig_param.shape} vs {onnx_param.shape}")
            
            # 2. Специальная обработка для attention слоев
            elif not transferred_this_param and 'mha' in onnx_name:
                transferred_this_param = self._transfer_attention_weights(
                    onnx_name, onnx_param, original_state)
                if transferred_this_param:
                    transferred += 1
            
            # 3. Поиск по схожести имен
            elif not transferred_this_param:
                orig_name = self._find_similar_param(onnx_name, original_state)
                if orig_name:
                    orig_param = original_state[orig_name]
                    if orig_param.shape == onnx_param.shape:
                        onnx_param.data.copy_(orig_param.data.cpu())
                        transferred += 1
                        transferred_this_param = True
                        print(f"Mapped {onnx_name} <- {orig_name}")
            
            if not transferred_this_param:
                print(f"No match found for {onnx_name} (shape: {onnx_param.shape})")
        
        print(f"Successfully transferred {transferred}/{total_params} parameters")
        
        # Проверяем что основные слои перенесены (исключаем value_net)
        key_layers = ['self_enc', 'ally_enc', 'enemy_enc', 'head_target', 'head_move_mu', 'head_aim_mu', 'head_fire_logit']
        for layer in key_layers:
            found = any(layer in name for name in onnx_state.keys())
            if found:
                print(f"✓ {layer} weights transferred")
            else:
                print(f"⚠ {layer} weights may not be transferred")
    
    def _transfer_attention_weights(self, onnx_name, onnx_param, original_state):
        """Специальная обработка для attention весов"""
        if 'w_o.weight' in onnx_name:
            orig_name = onnx_name.replace('mha.w_o.weight', 'mha.out_proj.weight')
            if orig_name in original_state and original_state[orig_name].shape == onnx_param.shape:
                onnx_param.data.copy_(original_state[orig_name].data.cpu())
                return True
                
        elif 'w_o.bias' in onnx_name:
            orig_name = onnx_name.replace('mha.w_o.bias', 'mha.out_proj.bias')
            if orig_name in original_state and original_state[orig_name].shape == onnx_param.shape:
                onnx_param.data.copy_(original_state[orig_name].data.cpu())
                return True
        
        elif 'w_q.weight' in onnx_name or 'w_k.weight' in onnx_name or 'w_v.weight' in onnx_name:
            orig_name = onnx_name.replace('mha.w_q.weight', 'mha.in_proj_weight').replace('mha.w_k.weight', 'mha.in_proj_weight').replace('mha.w_v.weight', 'mha.in_proj_weight')
            if orig_name in original_state:
                in_proj = original_state[orig_name]
                d_model = onnx_param.shape[0]
                
                if 'w_q.weight' in onnx_name:
                    onnx_param.data.copy_(in_proj[:d_model].cpu())
                    return True
                elif 'w_k.weight' in onnx_name:
                    onnx_param.data.copy_(in_proj[d_model:2*d_model].cpu())
                    return True
                elif 'w_v.weight' in onnx_name:
                    onnx_param.data.copy_(in_proj[2*d_model:3*d_model].cpu())
                    return True
        
        return False
    
    def _find_similar_param(self, onnx_name, original_state):
        """Ищет похожий параметр в оригинальной модели"""
        onnx_parts = onnx_name.split('.')
        
        if len(onnx_parts) >= 2:
            suffix = '.'.join(onnx_parts[-2:])
            
            for orig_name in original_state.keys():
                if orig_name.endswith(suffix):
                    orig_parts = orig_name.split('.')
                    common_parts = set(onnx_parts[:-2]) & set(orig_parts[:-2])
                    if len(common_parts) > 0:
                        return orig_name
        
        return None

    @torch.no_grad()
    def forward(self,
                self_vec,
                allies, allies_mask,
                enemies, enemies_mask,
                enemy_action_mask):
        """
        ИСПРАВЛЕНО: Убран global_state - он нужен только для value function
        Экспортируем только политику
        """
        
        # Убеждаемся что все на CPU
        inputs_cpu = {
            "self": self_vec.cpu(),
            "allies": allies.cpu(),
            "allies_mask": allies_mask.cpu(),
            "enemies": enemies.cpu(),
            "enemies_mask": enemies_mask.cpu(),
            "enemy_action_mask": enemy_action_mask.cpu(),
        }
        
        # Создаем dummy global_state для forward pass модели
        batch_size = self_vec.shape[0]
        dummy_global_state = torch.zeros(batch_size, 64, dtype=torch.float32)  # Размер из модели
        inputs_cpu["global_state"] = dummy_global_state
        
        # Forward pass через ONNX модель (получаем только logits, не value)
        logits, _ = self.onnx_model({"obs": inputs_cpu}, [], None)
        logits = logits.cpu()
        
        # Декодируем как в оригинале
        idx = 0
        logits_t = logits[:, idx:idx+self.ne]; idx += self.ne
        mu_move  = logits[:, idx:idx+2];        idx += 2
        idx += 2  # log_std_move
        mu_aim   = logits[:, idx:idx+2];        idx += 2
        idx += 2  # log_std_aim
        logit_fr = logits[:, idx:idx+1];        idx += 1

        target = torch.argmax(logits_t, dim=-1).unsqueeze(-1).float()
        move = torch.tanh(mu_move)
        aim  = torch.tanh(mu_aim)
        fire = (logit_fr > 0).float()
        
        result = torch.cat([target, move, aim, fire], dim=-1)
        return result.cpu()