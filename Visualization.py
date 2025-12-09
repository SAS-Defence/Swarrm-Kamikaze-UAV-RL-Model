import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class SwarmVisualization:
    def __init__(self):
        self.colors = cm.get_cmap('tab20c')
        sns.set_style("whitegrid")

    def create_training_dashboard(self, history, current_episode=0):
        """Eğitim dashboard'u oluştur"""
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(4, 4, figure=fig)

        # 1. Ödül Trendleri
        ax1 = fig.add_subplot(gs[0, :2])
        if 'episode_rewards' in history and len(history['episode_rewards']) > 0:
            rewards = history['episode_rewards']
            episodes = list(range(len(rewards)))

            ax1.plot(episodes, rewards, 'b-', linewidth=1.5, alpha=0.7, label='Episode Ödülü')

            # Hareketli ve Kümülatif Ortalamalar
            # KULLANICI İSTEĞİ: "Genel Ortalama"
            cum_avg = pd.Series(rewards).expanding().mean()
            ax1.plot(episodes, cum_avg, 'r-', linewidth=2.5, label='Genel Ortalama')
            
            # Kısa vadeli trend (20 Ep) - Anlık durum için gerekli
            if len(rewards) >= 20:
                short_avg = pd.Series(rewards).rolling(window=20).mean()
                ax1.plot(episodes, short_avg, 'g--', linewidth=1.5, alpha=0.7, label='Trend (20 Ep)')

            ax1.set_title('Episode Ödülleri & Genel Gelişim', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Toplam Ödül')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            if len(rewards) > 0:
                ax1.plot(len(rewards) - 1, rewards[-1], 'ro', markersize=8)

        # 2. Başarı Oranı (YENİ)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'success_rate' in history and len(history['success_rate']) > 0:
            success = history['success_rate']
            episodes = range(len(success))

            ax2.plot(episodes, success, 'g-', linewidth=2, alpha=0.3, label='Anlık')
            
            # Kümülatif Ortalama (Genel Başarı)
            cum_success = pd.Series(success).expanding().mean()
            ax2.plot(episodes, cum_success, 'darkgreen', linewidth=2.5, label='Genel Ort.')

            ax2.set_title('Başarı Oranı', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Başarı %')
            ax2.set_ylim(0, 105)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        # 3. Vuruş Oranı (Hit Rate) - KULLANICI İSTEĞİ (Epsilon yerine)
        ax3 = fig.add_subplot(gs[0, 3])
        if 'hit_rate' in history and len(history['hit_rate']) > 0:
            hits = history['hit_rate']
            ax3.plot(hits, 'orange', alpha=0.3, label='Anlık')
            
            # Kümülatif
            cum_hits = pd.Series(hits).expanding().mean()
            ax3.plot(cum_hits, 'darkorange', linewidth=2.5, label='Genel Ort.')
            
            ax3.set_title('Vuruş Başarısı\n(Hasar Veren Drone %)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Vuruş %')
            ax3.set_ylim(0, 105)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        # Fallback (Eski data varsa epsilon göster)
        elif 'epsilon' in history:
            epsilons = history['epsilon']
            ax3.plot(epsilons, 'gray', linewidth=2)
            ax3.set_title('Exploration Rate (ε)', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # 4. Hedef Başarı Oranları (YENİ - Target Breakdown)
        ax4 = fig.add_subplot(gs[1, :])
        target_types = ['tank', 'artillery', 'infantry', 'aircraft', 'radar']
        rates = []
        labels = []
        
        for t_type in target_types:
            key = f'{t_type}_rate'
            if key in history and len(history[key]) > 0:
                # KULLANICI İSTEĞİ: Tutarlılık için GENEL Ortalama kullan (Hasar grafiği ile aynı)
                # last_vals = history[key][-20:] # ESKİ: Sadece son 20
                avg_rate = np.mean(history[key]) # YENİ: Tüm Tarihçe
                rates.append(avg_rate)
                labels.append(f"{t_type}\n({avg_rate:.1f}%)")
            else:
                rates.append(0)
                labels.append(t_type)
        
        # Stacked Bar Chart (Destroyed + Damaged)
        colors_destroyed = ['darkgreen', 'brown', 'lightblue', 'gray', 'orange']
        colors_damaged = ['lightgreen', 'rosybrown', 'cyan', 'lightgray', 'yellow']
        
        rates_dmg = []
        for t_type in target_types:
            key_dmg = f'{t_type}_damage_rate'
            if key_dmg in history and len(history[key_dmg]) > 0:
                # KULLANICI İSTEĞİ: Son 20 değil, GENEL ortalama
                rates_dmg.append(np.mean(history[key_dmg])) 
            else:
                rates_dmg.append(0)

        # Plot Destroyed (Bottom)
        p1 = ax4.bar(labels, rates, color=colors_destroyed, alpha=0.9, label='İmha Edildi')
        # Plot Damaged (Top)
        p2 = ax4.bar(labels, rates_dmg, bottom=rates, color=colors_damaged, alpha=0.6, hatch='//', label='Hasar Aldı')

        ax4.set_title('Genel Hedef Başarı & Hasar Oranları (Tüm Episodelar)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Yüzde %')
        ax4.set_ylim(0, 100)
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend(loc='upper right')
        
        # Etiketler
        for i, (r_dest, r_dmg) in enumerate(zip(rates, rates_dmg)):
            total = r_dest + r_dmg
            if total > 0:
                ax4.text(i, total + 1, f'{total:.1f}%', ha='center', va='bottom', fontweight='bold')
            if r_dest > 0:
                ax4.text(i, r_dest/2, f'{r_dest:.1f}%', ha='center', va='center', color='white', fontsize=9)

        # 5. Loss Grafikleri (Actor & Critic Ayrıştırılmış)
        # Scaling sorunu nedeniyle (Critic çok küçük, Actor büyük) ayrı grafikler şart.
        
        # 5a. Actor Loss
        ax5 = fig.add_subplot(gs[2, :2])
        actor_keys = [k for k in history.keys() if 'actor_loss_drone_' in k]
        if actor_keys:
            # Sadece ilk drone'u göster (Kalabalık olmasın)
            key = actor_keys[0]
            if len(history[key]) > 0:
                losses = history[key]
                window = min(20, len(losses))
                if window > 0:
                    smooth = pd.Series(losses).rolling(window=window).mean()
                    episodes = list(range(len(losses)))
                    if len(smooth) == len(episodes):
                        ax5.plot(episodes, smooth, 'purple', label='Actor Loss (Policy)')
        
        ax5.set_title('Actor Loss (Pilot Kararlılığı)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Loss')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # 5b. Critic Loss
        ax6 = fig.add_subplot(gs[2, 2:])
        critic_keys = [k for k in history.keys() if 'critic_loss_drone_' in k]
        if critic_keys:
            key = critic_keys[0]
            if len(history[key]) > 0:
                losses = history[key]
                episodes = list(range(len(losses)))

                ax6.set_title('Critic Loss (Hoca Tahmin Hatası)', fontsize=12, fontweight='bold')
                ax6.set_ylabel('MSE Loss (Log Scale)')
                
                # 0 değerlerini temizle (Log scale hatasını önlemek için)
                safe_losses = [max(l, 1e-10) for l in losses]
                
                # Yeniden çiz
                ax6.clear()
                if len(episodes) == len(safe_losses):
                     # Raw Data (Daha silik)
                     ax6.plot(episodes, safe_losses, 'magenta', linewidth=0.5, alpha=0.3, label='Critic Loss (Raw)')
                     if len(safe_losses) > 20:
                          # Smoothed Data (Daha belirgin)
                          smooth_safe = pd.Series(safe_losses).rolling(window=min(20, len(safe_losses))).mean()
                          ax6.plot(episodes, smooth_safe, 'darkmagenta', linewidth=2.0, label='Trend')
                
                ax6.set_yscale('log')
                ax6.grid(True, alpha=0.3, which="both", ls="-") 
                ax6.legend()




        # 6. Actor ve Critic Loss Ayrımı
        ax6 = fig.add_subplot(gs[3, :2])
        actor_keys = [key for key in history.keys() if 'actor_loss_drone_' in key]
        critic_keys = [key for key in history.keys() if 'critic_loss_drone_' in key]

        if actor_keys and critic_keys:
            episodes = range(len(history[actor_keys[0]]))

            # İlk drone'un loss'larını al
            if len(history[actor_keys[0]]) > 0 and len(history[critic_keys[0]]) > 0:
                actor_losses = history[actor_keys[0]]
                critic_losses = history[critic_keys[0]]

                window = min(20, len(actor_losses))
                if window > 1:
                    actor_smooth = pd.Series(actor_losses).rolling(window=window).mean()
                    critic_smooth = pd.Series(critic_losses).rolling(window=window).mean()

                    lns1 = ax6.plot(episodes[window - 1:], actor_smooth[window - 1:],
                             'b-', linewidth=1.5, label='Actor Loss (Left)')
                    ax6.set_ylabel('Actor Loss', color='blue')
                    ax6.tick_params(axis='y', labelcolor='blue')

                    # İkinci Eksen (Critic Loss için)
                    ax6_twin = ax6.twinx()
                    
                    # 0 değerlerini temizle (Log scale için)
                    safe_critic_smooth = [max(x, 1e-10) for x in critic_smooth[window - 1:]]
                    
                    lns2 = ax6_twin.plot(episodes[window - 1:], safe_critic_smooth,
                             'r-', linewidth=1.5, label='Critic Loss (Right - Log)')
                    
                    ax6_twin.set_ylabel('Critic Loss (Log Scale)', color='red')
                    ax6_twin.tick_params(axis='y', labelcolor='red')
                    ax6_twin.set_yscale('log')

                    ax6.set_title('Actor vs Critic Loss (Drone 0)', fontsize=14, fontweight='bold')
                    ax6.set_xlabel('Öğrenme Adımı')
                    
                    # Ortak Lejant
                    lns = lns1 + lns2
                    labs = [l.get_label() for l in lns]
                    ax6.legend(lns, labs, loc='upper center')
                    
                    ax6.grid(True, alpha=0.3)

        # 7. Performans Metrikleri
        ax7 = fig.add_subplot(gs[3, 2:])
        ax7.axis('off')

        # Özet metni
        summary_text = "EĞİTİM ÖZETİ\n"
        summary_text += "=" * 40 + "\n\n"

        if 'episode_rewards' in history and len(history['episode_rewards']) > 0:
            rewards = history['episode_rewards']

            summary_text += f"Toplam Episode: {len(rewards)}\n"
            summary_text += f"Son Episode Ödülü: {rewards[-1]:.2f}\n"
            summary_text += f"Ortalama Ödül (son 10): {np.mean(rewards[-10:]):.2f}\n"
            summary_text += f"En İyi Ödül: {np.max(rewards):.2f}\n\n"

            if 'success_rate' in history and len(history['success_rate']) > 0:
                success = history['success_rate']
                summary_text += f"Son Başarı Oranı: {success[-1]:.1f}%\n"
                summary_text += f"Ortalama Başarı: {np.mean(success):.1f}%\n\n"

            if 'episode_length' in history and len(history['episode_length']) > 0:
                lengths = history['episode_length']
                summary_text += f"Ort. Episode Uzunluğu: {np.mean(lengths):.0f} adım\n"

            if 'epsilon' in history:
                summary_text += f"Exploration Rate: {history['epsilon'][-1]:.3f}\n"

        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.suptitle(f'FPV Kamikaze Sürüsü RL Eğitim Dashboard - Episode {current_episode}',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        return fig

    def create_performance_comparison(self, history):
        """Performans karşılaştırma grafikleri"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Ödül Dağılımı
        if 'episode_rewards' in history:
            rewards = history['episode_rewards']

            # Histogram
            axes[0, 0].hist(rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(np.mean(rewards), color='red', linestyle='--',
                               linewidth=2, label=f'Ortalama: {np.mean(rewards):.2f}')
            axes[0, 0].axvline(np.median(rewards), color='green', linestyle=':',
                               linewidth=2, label=f'Medyan: {np.median(rewards):.2f}')
            axes[0, 0].set_title('Ödül Dağılımı Histogramı')
            axes[0, 0].set_xlabel('Ödül')
            axes[0, 0].set_ylabel('Frekans')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Başarı Oranı Trendi
        if 'success_rate' in history:
            success = history['success_rate']
            episodes = range(len(success))

            axes[0, 1].plot(episodes, success, 'g-', linewidth=2)
            axes[0, 1].fill_between(episodes, success, alpha=0.3, color='green')

            # Trend çizgisi (Robust)
            if len(success) > 3:  # En az 4 nokta gerekli
                try:
                     z = np.polyfit(episodes, success, 3)
                     p = np.poly1d(z)
                     axes[0, 1].plot(episodes, p(episodes), "r--", alpha=0.7, linewidth=1.5)
                except Exception:
                     pass  # Hata olursa çizme

            axes[0, 1].set_title('Başarı Oranı Trendi')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Başarı %')
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Correlation Heatmap
        if 'episode_rewards' in history and 'success_rate' in history:
            metrics_to_correlate = {}

            for metric in ['episode_rewards', 'success_rate', 'episode_length', 'epsilon']:
                if metric in history and len(history[metric]) > 0:
                    metrics_to_correlate[metric] = history[metric][:100]  # İlk 100

            if len(metrics_to_correlate) >= 2:
                # Aynı uzunlukta yap
                min_len = min(len(v) for v in metrics_to_correlate.values())
                aligned_data = {k: v[:min_len] for k, v in metrics_to_correlate.items()}

                corr_df = pd.DataFrame(aligned_data)
                corr_matrix = corr_df.corr()

                im = axes[0, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

                # Değerleri yaz
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        text = axes[0, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                               ha='center', va='center', color='white')

                axes[0, 2].set_xticks(range(len(corr_matrix.columns)))
                axes[0, 2].set_yticks(range(len(corr_matrix.columns)))
                axes[0, 2].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                axes[0, 2].set_yticklabels(corr_matrix.columns)
                axes[0, 2].set_title('Metrikler Arası Korelasyon')

                plt.colorbar(im, ax=axes[0, 2])

        # 4. Learning Progress
        if 'episode_rewards' in history:
            rewards = history['episode_rewards']

            # Cumulative reward
            cumulative_rewards = np.cumsum(rewards)
            axes[1, 0].plot(cumulative_rewards, 'b-', linewidth=2)
            axes[1, 0].set_title('Kümülatif Ödül')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Toplam Ödül')
            axes[1, 0].grid(True, alpha=0.3)

            # Learning rate (tahmini)
            if len(rewards) > 10:
                windows = [5, 10, 20]
                colors = ['red', 'green', 'orange']

                for i, window in enumerate(windows):
                    if len(rewards) >= window:
                        moving_avg = pd.Series(rewards).rolling(window=window).mean()
                        axes[1, 1].plot(range(window - 1, len(rewards)), moving_avg[window - 1:],
                                        color=colors[i], linewidth=2, label=f'{window} Ep. Ort.')

                axes[1, 1].set_title('Ödül Trendleri (Hareketli Ortalama)')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Ortalama Ödül')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        # 5. Exploration vs Exploitation
        if 'epsilon' in history and 'episode_rewards' in history:
            epsilons = history['epsilon']
            rewards = history['episode_rewards']

            # Normalize rewards for comparison
            if len(rewards) > 1:
                rewards_norm = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)

                axes[1, 2].plot(epsilons, 'r-', linewidth=2, label='Exploration (ε)')
                axes[1, 2].plot(rewards_norm, 'b-', linewidth=2, alpha=0.5, label='Normalize Ödül')
                axes[1, 2].set_title('Exploration vs Ödül')
                axes[1, 2].set_xlabel('Episode')
                axes[1, 2].set_ylabel('Değer')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle('Performans Analizi ve Karşılaştırmalar', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def create_interactive_dashboard(self, history):
        """Plotly ile interaktif dashboard"""
        if 'episode_rewards' not in history or len(history['episode_rewards']) == 0:
            return None

        # Subplot oluştur
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Episode Ödülleri', 'Başarı Oranı', 'Exploration Rate',
                            'Ödül Dağılımı', 'Loss Trendleri', 'Actor vs Critic Loss',
                            'Hedef Başarı Oranları', 'Episode Uzunluğu', 'Kümülatif Ödül'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 1. Episode Ödülleri
        rewards = history['episode_rewards']
        episodes = list(range(len(rewards)))

        fig.add_trace(
            go.Scatter(x=episodes, y=rewards, mode='lines', name='Ödül',
                       line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # Hareketli ortalama
        if len(rewards) > 10:
            window = min(10, len(rewards))
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(x=episodes[window - 1:], y=moving_avg[window - 1:],
                           mode='lines', name=f'{window} Ep. Ort.',
                           line=dict(color='red', width=2)),
                row=1, col=1
            )

        # 2. Başarı Oranı
        if 'success_rate' in history:
            success = history['success_rate']
            fig.add_trace(
                go.Scatter(x=episodes[:len(success)], y=success, mode='lines',
                           name='Başarı %', line=dict(color='green', width=2)),
                row=1, col=2
            )

        # 3. Exploration Rate
        if 'epsilon' in history:
            epsilon = history['epsilon']
            fig.add_trace(
                go.Scatter(x=episodes[:len(epsilon)], y=epsilon, mode='lines',
                           name='ε', line=dict(color='orange', width=2)),
                row=1, col=3
            )

        # 4. Ödül Dağılımı
        fig.add_trace(
            go.Histogram(x=rewards, nbinsx=20, name='Ödül Dağılımı',
                         marker_color='blue', opacity=0.7),
            row=2, col=1
        )

        # 5. Loss Trendleri
        loss_keys = [k for k in history.keys() if 'loss_drone_' in k]
        if loss_keys and len(history[loss_keys[0]]) > 0:
            loss_episodes = list(range(len(history[loss_keys[0]])))

            for key in loss_keys[:3]:  # İlk 3 drone
                drone_id = key.split('_')[-1]
                losses = history[key]

                if len(losses) > 0:
                    fig.add_trace(
                        go.Scatter(x=loss_episodes[:len(losses)], y=losses,
                                   mode='lines', name=f'Drone {drone_id} Loss',
                                   line=dict(width=1.5)),
                        row=2, col=2
                    )

        # 6. Actor vs Critic Loss (INTERACTIVE EKLENDİ)
        actor_keys = [k for k in history.keys() if 'actor_loss_drone_' in k]
        critic_keys = [k for k in history.keys() if 'critic_loss_drone_' in k]
        
        if actor_keys and critic_keys and len(history[actor_keys[0]]) > 0:
             actor_loss = history[actor_keys[0]]
             critic_loss = history[critic_keys[0]]
             episodes_loss = list(range(len(actor_loss)))
             
             # Smooth
             window = min(20, len(actor_loss))
             if window > 1:
                  actor_smooth = pd.Series(actor_loss).rolling(window=window).mean()
                  critic_smooth = pd.Series(critic_loss).rolling(window=window).mean()
                  
                  fig.add_trace(
                      go.Scatter(x=episodes_loss[window-1:], y=actor_smooth[window-1:],
                                 mode='lines', name='Actor Loss (Smooth)', line=dict(color='blue')),
                      row=2, col=3
                  )
                  fig.add_trace(
                      go.Scatter(x=episodes_loss[window-1:], y=critic_smooth[window-1:],
                                 mode='lines', name='Critic Loss (Smooth)', line=dict(color='red')),
                      row=2, col=3
                  )

        # 7. Episode Uzunluğu
        if 'episode_length' in history:
            lengths = history['episode_length']
            fig.add_trace(
                go.Scatter(x=episodes[:len(lengths)], y=lengths, mode='lines',
                           name='Episode Uzunluğu', line=dict(color='purple', width=2)),
                row=3, col=2
            )

        # 8. Kümülatif Ödül
        cumulative_rewards = np.cumsum(rewards)
        fig.add_trace(
            go.Scatter(x=episodes, y=cumulative_rewards, mode='lines',
                       name='Kümülatif Ödül', line=dict(color='darkblue', width=2)),
            row=3, col=3
        )

        # Layout ayarları
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="FPV Kamikaze Sürüsü - Interaktif Dashboard",
            title_font_size=20
        )

        return fig