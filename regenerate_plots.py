
import torch
import sys
import os
import matplotlib.pyplot as plt
from Visualization import SwarmVisualization

def regenerate_dashboard(model_path):
    """
    Var olan bir model dosyasından (.pth) geçmişi okur ve 
    GÜNCEL Visualization.py kodunu kullanarak grafikleri yeniden çizer.
    """
    print(f"📂 Model yükleniyor: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ HATA: Dosya bulunamadı!")
        return

    try:
        # Checkpoint'i yükle (PyTorch 2.6+ uyumluluğu için weights_only=False)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # History kontrolü
        if 'history' not in checkpoint:
            print("❌ HATA: Model dosyasında eğitim geçmişi (history) bulunamadı.")
            return

        history = checkpoint['history']
        episode = checkpoint.get('episode', 0)
        
        print(f"✅ Model yüklendi. Episode: {episode}")
        print(f"📊 Mevcut veri anahtarları: {list(history.keys())}")

        # --- DATA DEBUGGING ---
        if 'tank_rate' in history:
            h = history['tank_rate']
            print(f"\n🔍 DETAYLI ANALİZ (tank_rate):")
            print(f"   - Toplam Veri Uzunluğu: {len(h)}")
            if len(h) > 0:
                print(f"   - Genel Ortalama: {sum(h)/len(h):.2f}%")
                print(f"   - Son 10 Değer: {h[-10:]}")
                print(f"   - İlk 10 Değer: {h[:10]}")
                zeros = h.count(0)
                print(f"   - Sıfır Olan Bölümler: {zeros} adet (%{zeros/len(h)*100:.1f})")
        
        if 'aircraft_rate' in history:
            h = history['aircraft_rate']
            print(f"\n🔍 DETAYLI ANALİZ (aircraft_rate):")
            print(f"   - Toplam Veri Uzunluğu: {len(h)}")
            if len(h) > 0:
                print(f"   - Genel Ortalama: {sum(h)/len(h):.2f}%")
                print(f"   - Son 10 Değer: {h[-10:]}")
                nonzero = [x for x in h if x > 0]
                print(f"   - Pozitif Değer Sayısı: {len(nonzero)}")
        
        if 'episode_rewards' in history:
            r = history['episode_rewards']
            print(f"\n🔍 DETAYLI ANALİZ (Rewards):")
            print(f"   - Uzunluk: {len(r)}")
            if len(r) > 0:
                 print(f"   - Min: {min(r):.2f}, Max: {max(r):.2f}, Mean: {sum(r)/len(r):.2f}")

        # LOSS ANALİZİ
        c_keys = [k for k in history.keys() if 'critic_loss' in k]
        if c_keys:
             val = history[c_keys[0]]
             if len(val) > 0:
                 print(f"\n🔍 DETAYLI ANALİZ (Critic Loss - {c_keys[0]}):")
                 print(f"   - Min: {min(val):.6f}")
                 print(f"   - Max: {max(val):.6f}")
                 print(f"   - Mean: {sum(val)/len(val):.6f}")
                 print(f"   - İlk 5: {val[:5]}")
                 print(f"   - Son 5: {val[-5:]}")
        
        a_keys = [k for k in history.keys() if 'actor_loss' in k]
        if a_keys:
             val = history[a_keys[0]]
             if len(val) > 0:
                 print(f"\n🔍 DETAYLI ANALİZ (Actor Loss - {a_keys[0]}):")
                 print(f"   - Mean: {sum(val)/len(val):.6f}")
        # ----------------------

        # Görselleştiriciyi başlat
        viz = SwarmVisualization()
        
        # Dashboard oluştur
        print("🎨 Grafikler çiziliyor...")
        fig = viz.create_training_dashboard(history, current_episode=episode)
        
        # Kaydet
        save_name = f"replot_dashboard_ep{episode}.png"
        save_path = os.path.join(os.path.dirname(model_path), save_name)
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Başarılı! Yeni grafik kaydedildi:")
        print(f"   📄 {save_path}")
        
    except Exception as e:
        print(f"❌ Beklenmedik bir hata oluştu: {e}")

if __name__ == "__main__":
    print("="*60)
    print("🔄 Angajman-RL Grafik Yenileme Aracı")
    print("="*60)
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        regenerate_dashboard(model_path)
    else:
        print("\nKullanım:")
        print("python regenerate_plots.py <model_dosyası_yolu>")
        print("\nÖrnek:")
        print("python regenerate_plots.py swarm_training_results/models/model_episode_500.pth")
