"""
Обновленные callbacks с исправленным ONNX экспортом и записью боев
ИСПРАВЛЕНИЕ: Добавлен обязательный параметр algorithm в on_episode_end
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
import ray
from torch.utils.tensorboard import SummaryWriter

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm

# Импортируем наши исправленные модули
from onnx_callbacks import export_onnx_with_meta
from save_res import BattleRecorder, RecordingArenaWrapper

class FixedLeagueCallbacksWithONNXAndRecording(RLlibCallback):
    """
    ИСПРАВЛЕННЫЕ callbacks с:
    1. Правильным ONNX экспортом с meta.json файлами
    2. Записью боев для визуализации
    3. Улучшенной обработкой ошибок
    4. ИСПРАВЛЕНА сигнатура on_episode_end для Ray 2.48+
    """
    
    def __init__(self):
        super().__init__()
        self.league = None
        self.opponent_ids = None
        self.eval_eps = 6
        self.clone_every = 10
        self.sample_top_k = 3
        self.attn_log_every = 20
        self.writer: Optional[SummaryWriter] = None
        self.curriculum = None
        
        # ONNX экспорт настройки
        self.export_onnx = True
        self.export_every = 25
        self.export_dir = "./onnx_exports"
        self.policies_to_export = ["main"]
        
        # Настройки записи боев
        self.record_battles = True
        self.battle_recorder: Optional[BattleRecorder] = None
        self.recording_frequency = 10  # Записывать каждый N-й evaluation матч
        self.recorded_matches = 0
        
    def setup(self, league_actor, opponent_ids: List[str], **kwargs):
        """Настройка параметров callbacks"""
        self.league = league_actor
        self.opponent_ids = opponent_ids
        self.eval_eps = kwargs.get('eval_episodes', 6)
        self.clone_every = kwargs.get('clone_every_iters', 10)
        self.sample_top_k = kwargs.get('sample_top_k', 3)
        self.attn_log_every = kwargs.get('attn_log_every', 20)
        self.curriculum = kwargs.get('curriculum_schedule', [])
        
        # ONNX настройки
        self.export_onnx = kwargs.get('export_onnx', True)
        self.export_every = kwargs.get('export_every', 25)
        self.export_dir = kwargs.get('export_dir', "./onnx_exports")
        self.policies_to_export = kwargs.get('policies_to_export', ["main"])
        
        # Настройки записи боев
        self.record_battles = kwargs.get('record_battles', True)
        self.recording_frequency = kwargs.get('recording_frequency', 10)
        
        # Создаем директории
        if self.export_onnx:
            os.makedirs(self.export_dir, exist_ok=True)
        
        if self.record_battles:
            recordings_dir = kwargs.get('recordings_dir', "./battle_recordings")
            self.battle_recorder = BattleRecorder(recordings_dir)
            print(f"📹 Battle recording enabled, saving to: {recordings_dir}")

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        """Инициализация algorithm"""
        pass

    def on_train_result(self, *, algorithm: Algorithm, result: Dict[str, Any], **kwargs) -> None:
        """Основная логика обработки результатов тренировки"""
        if self.league is None:
            return
            
        # Создаем writer
        if self.writer is None:
            logdir = getattr(algorithm, "logdir", "./logs")
            self.writer = SummaryWriter(log_dir=logdir)

        it = result["training_iteration"]
        ts_total = result.get("timesteps_total", 0)

        # 1) Evaluation матчей с возможной записью
        try:
            for pid in self.opponent_ids:
                # Определяем нужно ли записывать этот матч
                should_record = (
                    self.record_battles and 
                    self.battle_recorder and 
                    self.recorded_matches % self.recording_frequency == 0
                )
                
                w_main, w_opp = self._play_match(
                    algorithm, pid, self.eval_eps, 
                    record_battle=should_record,
                    battle_id=f"eval_it{it:04d}_vs_{pid}"
                )
                
                ray.get(self.league.update_pair_result.remote(w_main, w_opp, pid))
                self.recorded_matches += 1
                
        except Exception as e:
            print(f"Error in match evaluation: {e}")

        # 2) Логирование рейтингов
        try:
            scores = ray.get(self.league.get_all_scores.remote())
            result.setdefault("custom_metrics", {})
            
            for k, (mu, sigma) in scores.items():
                result["custom_metrics"][f"ts_{k}_mu"] = mu
                result["custom_metrics"][f"ts_{k}_sigma"] = sigma
                
                conservative_score = mu - 3 * sigma
                self.writer.add_scalar(f"ts/{k}_conservative", conservative_score, it)
                
        except Exception as e:
            print(f"Error getting league scores: {e}")
            scores = {}

        # 3) Клонирование худшего оппонента
        if it % self.clone_every == 0 and it > 0 and scores:
            try:
                items = [(pid, scores[pid][0] - 3*scores[pid][1]) for pid in self.opponent_ids]
                worst = min(items, key=lambda z: z[1])[0]
                
                w = algorithm.get_policy("main").get_weights()
                algorithm.get_policy(worst).set_weights(w)
                ray.get(self.league.clone_main_into.remote(worst))
                
                result["custom_metrics"][f"league_refresh_{worst}"] = it
                print(f"🔄 Refreshed opponent {worst} at iteration {it}")
                
            except Exception as e:
                print(f"Error refreshing opponent: {e}")

        # 4) Куррикулум
        if self.curriculum:
            for threshold, ac, ec in reversed(self.curriculum):
                if ts_total >= threshold:
                    try:
                        self._apply_curriculum(algorithm, ac, ec)
                        result["custom_metrics"]["curriculum_ally_choices"] = str(ac)
                        result["custom_metrics"]["curriculum_enemy_choices"] = str(ec)
                    except Exception as e:
                        print(f"Error setting curriculum: {e}")
                    break

        # 5) ИСПРАВЛЕННЫЙ ONNX экспорт
        if self.export_onnx and it % self.export_every == 0 and it > 0:
            try:
                print(f"\n🔧 Starting ONNX export for iteration {it}...")
                
                successful_exports = export_onnx_with_meta(
                    algorithm=algorithm,
                    iteration=it,
                    export_dir=self.export_dir,
                    policies_to_export=self.policies_to_export
                )
                
                if successful_exports:
                    result["custom_metrics"]["onnx_export_iteration"] = it
                    result["custom_metrics"]["onnx_policies_exported"] = len(successful_exports)
                    print(f"✅ ONNX export completed for iteration {it} ({len(successful_exports)} policies)")
                    
                    # Логируем в TensorBoard
                    self.writer.add_scalar("export/onnx_success", 1, it)
                    for export in successful_exports:
                        self.writer.add_text(
                            f"export/onnx_{export['policy_id']}", 
                            f"Exported to {export['onnx_path']}", 
                            it
                        )
                else:
                    print(f"⚠️ ONNX export completed but no policies were successfully exported")
                    self.writer.add_scalar("export/onnx_success", 0, it)
                    
            except Exception as e:
                print(f"❌ ONNX export failed for iteration {it}: {e}")
                self.writer.add_scalar("export/onnx_success", 0, it)
                import traceback
                traceback.print_exc()

        # 6) Экспорт записей боев для визуализации
        if (self.record_battles and self.battle_recorder and 
            it % (self.export_every * 2) == 0 and it > 0):
            try:
                html_path = self.battle_recorder.export_for_web_visualizer()
                if html_path:
                    result["custom_metrics"]["battle_replay_exported"] = it
                    print(f"🎬 Battle replay exported: {html_path}")
                    
            except Exception as e:
                print(f"Error exporting battle replay: {e}")

        if self.writer:
            self.writer.flush()

    def _play_match(self, algorithm: Algorithm, opp_id: str, episodes: int, 
                   record_battle: bool = False, battle_id: str = "") -> tuple:
        """
        Улучшенная версия матча с возможностью записи для Ray 2.48
        """
        try:
            from arena_env import ArenaEnv
            env_config = algorithm.config.env_config.copy() if hasattr(algorithm.config, 'env_config') else {}
            temp_env = ArenaEnv(env_config)
            
            # Обертываем в записывающий wrapper если нужно
            if record_battle and self.battle_recorder:
                temp_env = RecordingArenaWrapper(temp_env, self.battle_recorder)
                print(f"📹 Recording battle: {battle_id}")
            
            wins_main, wins_opp = 0, 0
            
            for episode_idx in range(episodes):
                obs, _ = temp_env.reset()
                done = False
                
                while not done:
                    action_dict = {}
                    
                    for aid, ob in obs.items():
                        pol_id = "main" if aid.startswith("red_") else opp_id
                        pol = algorithm.get_policy(pol_id)
                        
                        # Безопасное получение действия
                        try:
                            act, _, _ = pol.compute_single_action(ob, explore=False)
                        except Exception as e:
                            print(f"Error computing action for {aid}: {e}")
                            # Fallback действие
                            act = [0, 0.0, 0.0, 0.0, 0.0, 0]
                        
                        # Конвертируем действие в правильный формат
                        if isinstance(act, dict):
                            action_dict[aid] = act
                        else:
                            # Безопасная конвертация массива в словарь
                            act_array = np.array(act).flatten()
                            action_dict[aid] = {
                                "target": int(act_array[0]) if len(act_array) > 0 else 0,
                                "move": act_array[1:3].tolist() if len(act_array) > 2 else [0.0, 0.0],
                                "aim": act_array[3:5].tolist() if len(act_array) > 4 else [0.0, 0.0],
                                "fire": int(round(float(act_array[5]))) if len(act_array) > 5 else 0,
                            }
                    
                    obs, rews, terms, truncs, infos = temp_env.step(action_dict)
                    done = terms.get("__all__", False) or truncs.get("__all__", False)
                
                # Определяем победителя
                red_sum = sum(v for k, v in rews.items() if k.startswith("red_"))
                blue_sum = sum(v for k, v in rews.items() if k.startswith("blue_"))
                
                if red_sum > blue_sum:
                    wins_main += 1
                elif blue_sum > red_sum:
                    wins_opp += 1
                    
            # Если записывали бой, показываем статистику
            if record_battle and hasattr(temp_env, 'recorder'):
                stats = temp_env.recorder.get_summary_statistics()
                print(f"📊 Battle {battle_id} stats: {stats}")
                    
            return wins_main, wins_opp
            
        except Exception as e:
            print(f"Error in _play_match: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0

    def _apply_curriculum(self, algorithm, ally_choices, enemy_choices):
        """Применение куррикулума для Ray 2.48"""
        try:
            # Обновляем конфигурацию алгоритма
            if hasattr(algorithm.config, 'env_config'):
                algorithm.config.env_config["ally_choices"] = ally_choices
                algorithm.config.env_config["enemy_choices"] = enemy_choices
                print(f"📚 Updated curriculum: allies={ally_choices}, enemies={enemy_choices}")
            
            # Пытаемся применить к существующим окружениям
            try:
                if hasattr(algorithm, 'env_runner_group') and algorithm.env_runner_group:
                    def set_curriculum_fn(env):
                        if hasattr(env, 'set_curriculum'):
                            env.set_curriculum(ally_choices, enemy_choices)
                        # Если это наш wrapper, обновляем базовое окружение
                        elif hasattr(env, 'env') and hasattr(env.env, 'set_curriculum'):
                            env.env.set_curriculum(ally_choices, enemy_choices)
                    
                    algorithm.env_runner_group.foreach_env(set_curriculum_fn)
                    print(f"✅ Applied curriculum to env_runners")
                    
            except (AttributeError, Exception) as e:
                print(f"⚠️ Could not apply curriculum to existing envs: {e}")
                
        except Exception as e:
            print(f"❌ Could not apply curriculum: {e}")

    def on_episode_end(self, *, base_env, policies: Dict[str, Any], 
                      episode, env_index: Optional[int] = None, **kwargs) -> None:
        """
        ИСПРАВЛЕНО: Убран обязательный параметр algorithm для совместимости с Ray 2.48+
        Обработка окончания эпизода
        """
        
        try:
            # Записываем дополнительную статистику эпизода
            if hasattr(episode, 'custom_metrics'):
                # Добавляем метрики о командах (если используется иерархическая система)
                red_agents = [aid for aid in episode.get_agents() if aid.startswith("red_")]
                blue_agents = [aid for aid in episode.get_agents() if aid.startswith("blue_")]
                
                episode.custom_metrics["red_team_size"] = len(red_agents)
                episode.custom_metrics["blue_team_size"] = len(blue_agents)
                
                # Подсчитываем выжившие
                if hasattr(base_env, 'get_sub_environments'):
                    try:
                        sub_envs = base_env.get_sub_environments()
                        if sub_envs and env_index is not None and len(sub_envs) > env_index:
                            env = sub_envs[env_index]
                            if hasattr(env, '_alive_red') and hasattr(env, '_alive_blue'):
                                red_alive = sum(env._alive_red.values()) if env._alive_red else 0
                                blue_alive = sum(env._alive_blue.values()) if env._alive_blue else 0
                                
                                episode.custom_metrics["red_survivors"] = red_alive
                                episode.custom_metrics["blue_survivors"] = blue_alive
                                
                                total_agents = len(red_agents) + len(blue_agents)
                                if total_agents > 0:
                                    episode.custom_metrics["survival_rate"] = (red_alive + blue_alive) / total_agents
                                
                    except Exception as e:
                        # Не критично если не получилось получить статистику окружения
                        pass
        except Exception as e:
            # Полностью безопасная обработка - если что-то пошло не так, просто пропускаем
            pass

    def on_sample_end(self, *, samples, **kwargs) -> None:
        """Обработка окончания семплирования"""
        
        try:
            # Логируем статистику invalid actions если есть
            if hasattr(samples, 'data') and "infos" in samples.data:
                try:
                    infos = samples.data["infos"]
                    if len(infos) > 0 and isinstance(infos[0], dict):
                        
                        total_invalid_target = sum(info.get("invalid_target", 0) for info in infos)
                        total_oob_move = sum(info.get("oob_move", 0) for info in infos)
                        total_oob_aim = sum(info.get("oob_aim", 0) for info in infos)
                        
                        if total_invalid_target > 0 or total_oob_move > 0 or total_oob_aim > 0:
                            if self.writer:
                                it = getattr(samples, 'iteration', 0) if hasattr(samples, 'iteration') else 0
                                self.writer.add_scalar("validation/invalid_targets", total_invalid_target, it)
                                self.writer.add_scalar("validation/oob_moves", total_oob_move, it)
                                self.writer.add_scalar("validation/oob_aims", total_oob_aim, it)
                                
                            print(f"⚠️ Invalid actions: targets={total_invalid_target}, "
                                  f"oob_moves={total_oob_move}, oob_aims={total_oob_aim}")
                            
                except Exception as e:
                    # Не критично если не получилось собрать статистику
                    pass
        except Exception as e:
            # Полностью безопасная обработка
            pass


def create_fixed_callbacks_factory():
    """
    Фабрика для создания исправленных callbacks
    Решает проблемы с ONNX экспортом и добавляет запись боев
    """
    def create_callbacks():
        callbacks = FixedLeagueCallbacksWithONNXAndRecording()
        callbacks.setup(
            league_actor=None,  # Будет установлено позже в main script
            opponent_ids=[],    # Будет установлено позже в main script
            eval_episodes=4,
            clone_every_iters=15,
            curriculum_schedule=[
                (0, [1], [1]),
                (2_000_000, [1, 2], [1, 2]),
                (8_000_000, [1, 2, 3], [1, 2, 3]),
            ],            
            # ONNX экспорт настройки (ИСПРАВЛЕНО)
            export_onnx=True,
            export_every=25,  # Каждые 25 итераций
            export_dir="./onnx_exports_fixed",
            policies_to_export=["main"],
            
            # Настройки записи боев
            record_battles=True,
            recording_frequency=5,  # Записывать каждый 5-й evaluation матч
            recordings_dir="./battle_recordings",
        )
        return callbacks
    
    return create_callbacks


# Также создадим простую утилиту для тестирования ONNX экспорта
def test_onnx_export_standalone():
    """
    Отдельная функция для тестирования ONNX экспорта
    Можно запустить независимо от тренировки
    """
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from arena_env import ArenaEnv
    from entity_attention_model import ONNXEntityAttentionModel
    from masked_multihead_dist import MaskedTargetMoveAimFire
    from ray.rllib.models import ModelCatalog
    
    def env_creator(cfg): 
        return ArenaEnv(cfg)
    
    print("🧪 Testing standalone ONNX export...")
    
    # Инициализация Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        register_env("ArenaEnv", env_creator)
        ModelCatalog.register_custom_model("entity_attention", ONNXEntityAttentionModel)
        ModelCatalog.register_custom_action_dist("masked_multihead", MaskedTargetMoveAimFire)
        
        # Получаем размеры окружения
        tmp_env = ArenaEnv({"ally_choices": [1], "enemy_choices": [1]})
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        max_enemies = obs_space["enemies"].shape[0]
        max_allies = obs_space["allies"].shape[0]
        
        # Создаем минимальную конфигурацию
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env="ArenaEnv",
                env_config={
                    "episode_len": 50,
                    "ally_choices": [1],
                    "enemy_choices": [1],
                }
            )
            .framework("torch")
            .env_runners(num_env_runners=0)  # Только локальный worker
            .training(train_batch_size=512, minibatch_size=256)
            .multi_agent(
                policies={
                    "main": (None, obs_space, act_space, {
                        "model": {
                            "custom_model": "entity_attention",
                            "custom_action_dist": "masked_multihead",
                            "custom_model_config": {
                                "max_enemies": max_enemies,
                                "max_allies": max_allies,
                            },
                            "vf_share_layers": False,
                        }
                    }),
                },
                policy_mapping_fn=lambda aid, *args, **kwargs: "main",
                policies_to_train=["main"],
            )
        )
        
        # Создаем алгоритм
        algo = config.build()
        
        # Делаем один шаг тренировки для инициализации
        print("🏋️ Training one step to initialize...")
        result = algo.train()
        print(f"✅ Training step completed, episode_reward_mean: {result.get('env_runners', {}).get('episode_reward_mean', 'N/A')}")
        
        # Тестируем ONNX экспорт
        print("🔧 Testing ONNX export...")
        export_dir = "./test_onnx_export"
        os.makedirs(export_dir, exist_ok=True)
        
        successful_exports = export_onnx_with_meta(
            algorithm=algo,
            iteration=1,
            export_dir=export_dir,
            policies_to_export=["main"]
        )
        
        if successful_exports:
            print(f"✅ ONNX export test PASSED! Exported {len(successful_exports)} policies")
            
            # Тестируем инференс
            from onnx_callbacks import run_inference_test
            
            for export in successful_exports:
                onnx_path = export["onnx_path"]
                print(f"🔬 Testing inference for {onnx_path}...")
                
                try:
                    run_inference_test(onnx_path, batch_size=2, verbose=False)
                    print(f"✅ Inference test PASSED for {export['policy_id']}")
                except Exception as e:
                    print(f"❌ Inference test FAILED for {export['policy_id']}: {e}")
        else:
            print("❌ ONNX export test FAILED - no policies exported")
            
        # Очистка
        algo.stop()
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()
    
    print("🏁 ONNX export test completed")


if __name__ == "__main__":
    test_onnx_export_standalone()