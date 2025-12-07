import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from Environment import SwarmBattlefield2D
from TrainLoop import HierarchicalSwarmTrainer
from Visualization import SwarmVisualization
from SwarmCoordinator import SwarmCoordinator  # YENİ!


class SwarmTrainingManager:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.env = None
        self.trainer = None
        self.coordinator = None  # YENİ!
        self.visualizer = SwarmVisualization()

        # Sonuç dizinleri
        self.base_dir = "swarm_training_results"
        self.create_directories()

        # Kayıtlar
        self.training_log = []
        self.start_time = None

        print("=" * 70)
        print("FPV KAMİKAZE SÜRÜSÜ RL EĞİTİM SİSTEMİ")
        print("🎯 SÜRÜ MUHAKEME VE KOORDİNASYON SİSTEMİ AKTİF")
        print("=" * 70)

    def _load_config(self, config_path):
        """Konfigürasyon yükle - GÜNCELLENMİŞ"""
        default_config = {
            # Ortam parametreleri
            'env': {
                'width': 1200,
                'height': 800,
                'num_drones': 6,
                'num_targets': 12,
                'max_steps': 1000
            },

            # GELİŞTİRİLMİŞ eğitim parametreleri
            'training': {
                'total_episodes': 1000,  # ARTIRILDI
                'batch_size': 64,  # AZALTILDI
                'gamma': 0.99,
                'learning_rate': 0.0001,  # AZALTILDI
                'epsilon_start': 1.0,
                'epsilon_end': 0.1,  # ARTIRILDI
                'epsilon_decay': 0.996,  # SÜPER YAVASLATILDI (Sabır Yaması)
                'tau': 0.001,  # YAVASLATILDI
                'buffer_size': 20000,  # ARTIRILDI
                'save_interval': 50,
                'eval_interval': 20,
                'render_interval': 100
            },

            # GELİŞTİRİLMİŞ curriculum learning
            'curriculum': {
                'enabled': True,
                'stages': [
                    # Çok kolay başla - Drone sayısı sabit, hedef artar
                    {'episodes': 300, 'num_targets': 3, 'width': 600, 'height': 400},
                    {'episodes': 300, 'num_targets': 6, 'width': 800, 'height': 600},
                    {'episodes': 250, 'num_targets': 9, 'width': 1000, 'height': 700},
                    {'episodes': 500, 'num_targets': 12, 'width': 1200, 'height': 800}
                ]
            },

            # 🎯 YENİ: Koordinasyon ayarları
            'coordination': {
                'enabled': True,  # Koordinatörü kullan
                'verbose': True  # Koordinatör loglarını göster
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                import copy
                merged_config = copy.deepcopy(default_config)
                self._deep_update(merged_config, user_config)
                return merged_config

        return default_config

    def _deep_update(self, target, source):
        """Deep dictionary update"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target:
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def create_directories(self):
        """Dizinleri oluştur"""
        dirs = ['models', 'logs', 'plots', 'dashboards', 'videos']
        for dir_name in dirs:
            os.makedirs(os.path.join(self.base_dir, dir_name), exist_ok=True)

    def setup_environment(self, stage_config=None):
        """Ortamı kur"""
        env_config = self.config['env'].copy()

        if stage_config:
            for k, v in stage_config.items():
                if k in ['width', 'height', 'num_targets', 'max_steps']:
                    env_config[k] = v

        self.env = SwarmBattlefield2D(
            width=env_config['width'],
            height=env_config['height'],
            num_drones=env_config['num_drones'],
            num_targets=env_config['num_targets']
        )

        self.env.max_steps = env_config.get('max_steps', 1000)

        # 🎯 KOORDİNATÖRÜ KURU
        if self.config['coordination']['enabled']:
            self.coordinator = SwarmCoordinator(self.env)
            print(f"[SETUP] ✅ Sürü koordinatörü aktif!")
        else:
            self.coordinator = None
            print(f"[SETUP] ⚠️  Koordinatör kapalı - bağımsız drone'lar")

        print(f"[SETUP] Ortam: {env_config['width']}x{env_config['height']}")
        print(f"[SETUP] {env_config['num_drones']} drone, {env_config['num_targets']} hedef")

    def setup_trainer(self):
        """Eğitmeni kur"""
        self.trainer = HierarchicalSwarmTrainer(
            self.env,
            self.config['training']
        )

        print(f"[SETUP] Eğitmen oluşturuldu")
        print(f"[SETUP] State dim: {self.trainer.state_dim} (koordinasyon dahil)")

    def evaluate_policy(self, num_episodes=5, render=False):
        """Politikayı değerlendir - KOORDİNASYONLU"""
        print(f"\n[EVAL] Politik değerlendirme ({num_episodes} episode)...")

        eval_rewards = []
        eval_success_rates = []

        for eval_ep in range(num_episodes):
            observations = self.env.reset()

            if self.coordinator:
                self.coordinator.reset()

            episode_reward = 0
            done = False

            if render:
                self.env.render()

            while not done:
                # Koordinatörden direktif al
                directives = None
                if self.coordinator:
                    directives = self.coordinator.get_strategic_actions(observations)

                actions = []

                for drone_id, obs in enumerate(observations):
                    if obs.get('health', 1.0) <= 0 or obs.get('battery', 0) <= 0:
                        actions.append([0.0, 0.0, 0, -1])
                        continue

                    # Direktif
                    directive = directives[drone_id] if directives else None

                    # State + direktif
                    state = self.trainer._process_observation(obs, directive)
                    action, _, _, _ = self.trainer.agents[drone_id].get_action(state, epsilon=0.0)

                    move_x = float(action[0])
                    move_y = float(action[1])
                    attack = int(action[2])
                    target_id = -1

                    # Koordinatörden hedef al
                    if directive and directive.get('target_id', -1) >= 0:
                        target_id = directive['target_id']
                        if directive.get('should_attack', False):
                            attack = 1
                    else:
                        # Fallback
                        if attack == 1:
                            visible = obs.get('visible_targets', [])
                            if isinstance(visible, list) and len(visible) > 0:
                                def target_key(t):
                                    imp = float(t.get('importance', 0.0))
                                    dist = float(t.get('distance', 1.0))
                                    return (imp, - (1.0 - dist))

                                best = max(visible, key=target_key)
                                target_id = int(best.get('id', -1))

                    actions.append([move_x, move_y, attack, target_id])

                observations, rewards, done, info = self.env.step(actions)
                episode_reward += sum(rewards)

                if render:
                    self.env.render()
                    time.sleep(0.01)

            eval_rewards.append(episode_reward)
            eval_success_rates.append(info.get('success_rate', 0))

            print(f"[EVAL] Episode {eval_ep + 1}: Ödül={episode_reward:.2f}, "
                  f"Başarı={info.get('success_rate', 0):.1f}%")

        avg_reward = np.mean(eval_rewards)
        avg_success = np.mean(eval_success_rates)

        print(f"[EVAL] Ortalama Ödül: {avg_reward:.2f}")
        print(f"[EVAL] Ortalama Başarı: {avg_success:.1f}%")

        return avg_reward, avg_success

    def run_training(self):
        """Ana eğitim döngüsü"""
        print("\n" + "=" * 70)
        print("EĞİTİM BAŞLIYOR")
        if self.config['coordination']['enabled']:
            print("🎯 SÜRÜ KOORDİNASYONU AKTİF")
        print("=" * 70)

        self.start_time = time.time()

        # Curriculum learning
        if self.config['curriculum']['enabled']:
            self._run_curriculum_training()
        else:
            self._run_standard_training()

        # Final değerlendirme
        print("\n" + "=" * 70)
        print("FINAL DEĞERLENDİRME")
        print("=" * 70)

        final_reward, final_success = self.evaluate_policy(num_episodes=10, render=False)

        # Final dashboard
        self.create_final_dashboard(final_reward, final_success)

        # Eğitim özeti
        self.print_training_summary()

    def _run_curriculum_training(self):
        """Curriculum learning ile eğitim"""
        stages = self.config['curriculum']['stages']
        total_episodes = 0
        fixed_num_drones = self.config['env']['num_drones']

        for stage_idx, stage in enumerate(stages):
            print(f"\n{'=' * 60}")
            print(f"AŞAMA {stage_idx + 1}/{len(stages)}")
            print(f"{'=' * 60}")
            print(f"Episode: {stage['episodes']}")
            print(f"Drone: {fixed_num_drones}, Hedef: {stage['num_targets']}")
            print(f"Harita: {stage['width']}x{stage['height']}")

            # Ortamı kur
            self.setup_environment(stage)

            # İlk aşamada trainer oluştur
            if stage_idx == 0:
                self.setup_trainer()
            else:
                self.trainer.env = self.env
                # Koordinatörü de güncelle
                if self.coordinator:
                    self.coordinator.env = self.env

            # Aşama eğitimi
            stage_start = self.trainer.episode
            stage_end = stage_start + stage['episodes']

            while self.trainer.episode < stage_end:
                self._train_single_episode()
                total_episodes += 1

            print(f"[STAGE] Aşama {stage_idx + 1} tamamlandı")

            # Aşama sonu değerlendirme
            if stage_idx < len(stages) - 1:
                avg_reward, avg_success = self.evaluate_policy(num_episodes=3)
                print(f"[STAGE] Aşama {stage_idx + 1} değerlendirmesi:")
                print(f"       Ortalama Ödül: {avg_reward:.2f}")
                print(f"       Ortalama Başarı: {avg_success:.1f}%")

    def _run_standard_training(self):
        """Standart eğitim"""
        self.setup_environment()
        self.setup_trainer()

        total_episodes = self.config['training']['total_episodes']

        for ep in range(total_episodes):
            self._train_single_episode()

    def _train_single_episode(self):
        """Tek episode eğit - KOORDİNASYONLU"""
        episode_num = self.trainer.episode + 1

        # 🎯 KOORDİNATÖRÜ KULLANARAK EĞİT
        episode_reward, info = self.trainer.train_episode(coordinator=self.coordinator)

        # Kayıt
        log_entry = {
            'episode': episode_num,
            'reward': episode_reward,
            'success_rate': info.get('success_rate', 0),
            'destroyed_targets': info.get('destroyed_targets', 0),
            'destroyed_drones': info.get('destroyed_drones', 0),
            'epsilon': self.trainer.epsilon,
            'timestamp': datetime.now().isoformat()
        }

        # Koordinatör bilgilerini ekle
        if self.coordinator:
            coord_summary = self.coordinator.get_mission_summary()
            log_entry['coordination'] = coord_summary

        self.training_log.append(log_entry)

        # Periyodik işlemler
        save_interval = self.config['training']['save_interval']
        eval_interval = self.config['training']['eval_interval']
        render_interval = self.config['training']['render_interval']

        # Model kaydet
        if episode_num % save_interval == 0:
            model_path = os.path.join(self.base_dir, 'models')
            self.trainer.save_model(model_path)

            # Log kaydet
            log_path = os.path.join(self.base_dir, 'logs', f'training_log_ep{episode_num}.json')
            with open(log_path, 'w') as f:
                json.dump(self.training_log, f, indent=2)

            print(f"[SAVE] Model ve log kaydedildi (Episode {episode_num})")

        # Değerlendirme
        if episode_num % eval_interval == 0:
            eval_reward, eval_success = self.evaluate_policy(num_episodes=2)

            # History'e kaydet
            self.trainer.history.setdefault('eval_rewards', []).append(eval_reward)
            self.trainer.history.setdefault('eval_success', []).append(eval_success)

            print(f"[EVAL] Episode {episode_num}: Eval Ödül={eval_reward:.2f}, "
                  f"Eval Başarı={eval_success:.1f}%")

        # Görselleştirme
        if episode_num % render_interval == 0:
            dashboard = self.visualizer.create_training_dashboard(
                self.trainer.history,
                current_episode=episode_num
            )

            dashboard_path = os.path.join(self.base_dir, 'dashboards',
                                          f'dashboard_ep{episode_num:04d}.png')
            dashboard.savefig(dashboard_path, dpi=150, bbox_inches='tight')
            plt.close(dashboard)

            print(f"[VIZ] Dashboard kaydedildi: {dashboard_path}")

    def create_final_dashboard(self, final_reward, final_success):
        """Final dashboard oluştur"""
        print("\n[FINAL] Final dashboard oluşturuluyor...")

        # 1. Training dashboard
        training_fig = self.visualizer.create_training_dashboard(
            self.trainer.history,
            current_episode=self.trainer.episode
        )

        training_path = os.path.join(self.base_dir, 'plots', 'training_dashboard.png')
        training_fig.savefig(training_path, dpi=200, bbox_inches='tight')
        plt.close(training_fig)

        # 2. Performance comparison
        comparison_fig = self.visualizer.create_performance_comparison(self.trainer.history)
        comparison_path = os.path.join(self.base_dir, 'plots', 'performance_comparison.png')
        comparison_fig.savefig(comparison_path, dpi=200, bbox_inches='tight')
        plt.close(comparison_fig)

        # 3. Interactive dashboard
        interactive_fig = self.visualizer.create_interactive_dashboard(self.trainer.history)
        if interactive_fig:
            interactive_path = os.path.join(self.base_dir, 'dashboards', 'interactive_dashboard.html')
            interactive_fig.write_html(interactive_path)

        # 4. Final summary
        self._create_final_summary_plot(final_reward, final_success)

        print(f"[FINAL] Dashboard'lar kaydedildi: {self.base_dir}/plots/")

    def _create_final_summary_plot(self, final_reward, final_success):
        """Final özet grafiği"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Ödül trendi
        if 'episode_rewards' in self.trainer.history:
            rewards = self.trainer.history['episode_rewards']
            episodes = range(len(rewards))

            axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.5, linewidth=1)

            if len(rewards) > 10:
                window = min(50, len(rewards))
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                axes[0, 0].plot(episodes[window - 1:], moving_avg[window - 1:],
                                'r-', linewidth=2, label=f'{window} Ep. Ort.')

            axes[0, 0].set_title('Öğrenme Eğrisi', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Ödül')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Başarı oranı
        if 'success_rate' in self.trainer.history:
            success = self.trainer.history['success_rate']
            episodes = range(len(success))

            axes[0, 1].plot(episodes, success, 'g-', linewidth=2)
            axes[0, 1].fill_between(episodes, success, alpha=0.3, color='green')
            axes[0, 1].axhline(y=final_success, color='r', linestyle='--',
                               linewidth=2, label=f'Final: {final_success:.1f}%')

            axes[0, 1].set_title('Başarı Oranı Gelişimi', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Başarı %')
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Hedef başarı oranları
        target_types = ['tank', 'artillery', 'infantry', 'aircraft', 'radar']
        target_colors = ['darkgreen', 'brown', 'lightblue', 'gray', 'orange']

        final_rates = []
        for target_type in target_types:
            rate_key = f'{target_type}_rate'
            if rate_key in self.trainer.history and len(self.trainer.history[rate_key]) > 0:
                final_rates.append(self.trainer.history[rate_key][-1])
            else:
                final_rates.append(0)

        bars = axes[1, 0].bar(target_types, final_rates, color=target_colors, alpha=0.8)
        axes[1, 0].set_title('Hedef Tipi Başarı Oranları', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Başarı %')
        axes[1, 0].set_ylim(0, 100)

        for bar, rate in zip(bars, final_rates):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 2,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

        # 4. Özet metni
        axes[1, 1].axis('off')

        training_time = time.time() - self.start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)

        summary_text = "EĞİTİM ÖZETİ\n"
        summary_text += "=" * 40 + "\n\n"

        if self.config['coordination']['enabled']:
            summary_text += "🎯 SÜRÜ KOORDİNASYONU: AKTİF\n\n"

        summary_text += f"Toplam Episode: {self.trainer.episode}\n"
        summary_text += f"Toplam Adım: {self.trainer.total_steps}\n"
        summary_text += f"Eğitim Süresi: {hours:02d}:{minutes:02d}:{seconds:02d}\n\n"

        if 'episode_rewards' in self.trainer.history:
            rewards = self.trainer.history['episode_rewards']
            summary_text += f"Ortalama Ödül: {np.mean(rewards):.2f}\n"
            summary_text += f"En İyi Ödül: {np.max(rewards):.2f}\n"
            summary_text += f"Son Ödül: {rewards[-1]:.2f}\n\n"

        summary_text += f"Final Ödül: {final_reward:.2f}\n"
        summary_text += f"Final Başarı: {final_success:.1f}%\n\n"

        summary_text += "PERFORMANS:\n"
        if final_success > 70:
            summary_text += "✅ MÜKEMMEL! Sürü koordineli çalışıyor.\n"
        elif final_success > 40:
            summary_text += "👍 İYİ! Koordinasyon gelişiyor.\n"
        elif final_success > 15:
            summary_text += "⚠️  ORTA! Daha fazla eğitim gerekli.\n"
        else:
            summary_text += "❌ DÜŞÜK! Parametre ayarı gerekli.\n"

        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('FPV Kamikaze Sürüsü - Koordinasyonlu RL Eğitimi',
                     fontsize=18, fontweight='bold', y=1.02)

        plt.tight_layout()

        final_plot_path = os.path.join(self.base_dir, 'plots', 'final_summary.png')
        plt.savefig(final_plot_path, dpi=200, bbox_inches='tight')
        plt.close()

    def print_training_summary(self):
        """Eğitim özetini yazdır"""
        print("\n" + "=" * 70)
        print("EĞİTİM ÖZETİ")
        print("=" * 70)

        training_time = time.time() - self.start_time
        hours = int(training_time // 3600)
        minutes = int(training_time % 3600 // 60)
        seconds = int(training_time % 60)

        print(f"\n⏱️  Toplam Süre: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"📊 Toplam Episode: {self.trainer.episode}")
        print(f"👣 Toplam Adım: {self.trainer.total_steps}")

        if 'episode_rewards' in self.trainer.history:
            rewards = self.trainer.history['episode_rewards']
            print(f"\n💰 Ödül İstatistikleri:")
            print(f"   Ortalama: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"   En İyi: {np.max(rewards):.2f}")
            print(f"   Son: {rewards[-1]:.2f}")

        if 'success_rate' in self.trainer.history:
            success = self.trainer.history['success_rate']
            print(f"\n🎯 Başarı İstatistikleri:")
            print(f"   Ortalama: {np.mean(success):.1f}%")
            print(f"   En İyi: {np.max(success):.1f}%")
            print(f"   Son: {success[-1]:.1f}%")

        if self.config['coordination']['enabled']:
            print(f"\n🎯 Koordinasyon: AKTİF ✅")

        print(f"\n📁 Sonuçlar: {self.base_dir}/")
        print(f"   📊 Grafikler: {self.base_dir}/plots/")
        print(f"   💾 Modeller: {self.base_dir}/models/")
        print(f"   📝 Loglar: {self.base_dir}/logs/")

        print("\n" + "=" * 70)
        print("EĞİTİM TAMAMLANDI! 🎉")
        print("=" * 70)


def main():
    """Ana fonksiyon"""
    print("FPV Kamikaze Sürüsü RL Eğitim Sistemi")
    print("Version: 3.0 - Sürü Koordinasyonu")

    config_path = None

    # Training manager oluştur
    manager = SwarmTrainingManager(config_path)

    # Eğitimi başlat
    try:
        manager.run_training()
    except KeyboardInterrupt:
        print("\n\n⚠️  Eğitim durduruldu!")
        print("   Model kaydediliyor...")

        if manager.trainer:
            model_path = os.path.join(manager.base_dir, 'models', 'interrupted_model.pth')
            manager.trainer.save_model(os.path.dirname(model_path))

            if manager.trainer.history:
                dashboard = manager.visualizer.create_training_dashboard(
                    manager.trainer.history,
                    current_episode=manager.trainer.episode
                )
                dashboard_path = os.path.join(manager.base_dir, 'plots', 'interrupted_dashboard.png')
                dashboard.savefig(dashboard_path, dpi=150, bbox_inches='tight')
                plt.close(dashboard)

        print(f"   Kaydedildi: {manager.base_dir}/")
    except Exception as e:
        print(f"\n\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()