import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class SwarmCoordinator:
    """
    Sürü Muhakeme ve Koordinasyon Sistemi

    Görevler:
    1. Hedefleri önem sırasına göre önceliklendirme
    2. Her hedefe optimal sayıda drone atama
    3. Hedef yok edilince yeni hedef atama
    4. Koordineli saldırı yönetimi
    """

    def __init__(self, env):
        self.env = env

        # Önem sıralaması (yüksekten düşüğe)
        self.target_priority = {
            'aircraft': 100,  # Hava tehdidi en önemli
            'tank': 90,  # Ağır zırhlı
            'artillery': 80,  # Uzun menzilli tehdit
            'radar': 70,  # Elektronik savaş
            'infantry': 50  # Yumuşak hedef
        }

        # Hedef atamaları: {target_id: [drone_ids]}
        self.target_assignments = defaultdict(list)

        # Drone durumları: {drone_id: {'status', 'target', 'role'}}
        self.drone_states = {}

        # Mission log
        self.mission_log = []

        print("[COORDINATOR] Sürü koordinatörü aktif")
        print(
            f"[COORDINATOR] Hedef öncelik sırası: {sorted(self.target_priority.items(), key=lambda x: x[1], reverse=True)}")

    def reset(self):
        """Koordinatörü sıfırla"""
        self.target_assignments.clear()
        self.drone_states.clear()
        self.mission_log.clear()

        # Tüm drone'ları serbest duruma al
        for drone in self.env.drones:
            self.drone_states[drone['id']] = {
                'status': 'idle',
                'target': None,
                'role': 'hunter',
                'last_action': None
            }

    def get_strategic_actions(self, observations):
        """
        ANA MUHAKEME FONKSİYONU

        Her adımda:
        1. Hedef durumlarını analiz et
        2. Atamaları güncelle
        3. Her drone için stratejik direktif üret

        Return: List[Dict] - Her drone için stratejik direktif
        """
        # 1. Hedef durumunu analiz et
        available_targets = self._analyze_targets()

        # 2. Kritik atamaları kontrol et ve güncelle
        self._update_assignments(available_targets)

        # 3. Serbest drone'ları ata
        self._assign_idle_drones(available_targets)

        # 4. Her drone için direktif oluştur
        directives = []
        for drone_id, obs in enumerate(observations):
            directive = self._generate_directive(drone_id, obs, available_targets)
            directives.append(directive)

        return directives

    def _analyze_targets(self):
        """
        Hedefleri analiz et ve öncelik sırasına koy

        Return: List[Dict] - Önceliklendirilmiş hedef listesi
        """
        targets_info = []

        for target in self.env.targets:
            if target['destroyed']:
                continue

            # Önem puanı hesapla
            base_priority = self.target_priority.get(target['type'], 50)

            # Faktörler:
            # - HP düşükse öncelik artar (bitirmek kolay)
            # - Zaten saldırı altındaysa öncelik azalır
            # - Tespit edilmemişse öncelik azalır

            hp_factor = 1.0 - (target['hp'] / target['max_hp'])  # 0-1, düşük HP = yüksek faktör
            detection_factor = 1.0 if target['detected'] else 0.5

            # Mevcut saldırgan sayısı
            current_attackers = len(target['attackers'])
            required = target['required_drones']

            # Eğer yeterli saldırgan varsa, önceliği düşür
            if current_attackers >= required:
                attack_factor = 0.3
            elif current_attackers > 0:
                attack_factor = 0.7  # Yarı tamamlanmış, destek gerekebilir
            else:
                attack_factor = 1.0  # Kimse saldırmıyor, yüksek öncelik

            # Final öncelik puanı
            priority_score = base_priority * detection_factor * attack_factor * (1.0 + hp_factor)

            targets_info.append({
                'id': target['id'],
                'type': target['type'],
                'priority_score': priority_score,
                'base_priority': base_priority,
                'position': (target['x'], target['y']),
                'hp': target['hp'],
                'max_hp': target['max_hp'],
                'required_drones': required,
                'current_attackers': current_attackers,
                'detected': target['detected'],
                'needs_support': current_attackers < required and current_attackers > 0
            })

        # Öncelik puanına göre sırala (yüksekten düşüğe)
        targets_info.sort(key=lambda x: x['priority_score'], reverse=True)

        return targets_info

    def _update_assignments(self, available_targets):
        """
        Mevcut atamaları güncelle

        - Hedef yok edildiyse, o drone'ları serbest bırak
        - Yetersiz drone varsa, destek çağır
        """
        target_ids = {t['id'] for t in available_targets}

        # Yok edilmiş hedeflere atanmış drone'ları serbest bırak
        assignments_to_remove = []

        for target_id, drone_ids in self.target_assignments.items():
            if target_id not in target_ids:
                # Hedef yok edilmiş
                print(f"[COORDINATOR] ✅ Hedef {target_id} imha edildi! Drone'lar {drone_ids} serbest bırakılıyor")

                for drone_id in drone_ids:
                    if drone_id in self.drone_states:
                        self.drone_states[drone_id]['status'] = 'idle'
                        self.drone_states[drone_id]['target'] = None

                assignments_to_remove.append(target_id)

        # Temizle
        for target_id in assignments_to_remove:
            del self.target_assignments[target_id]

        # Destek gereksinimi kontrolü
        for target_info in available_targets:
            target_id = target_info['id']
            required = target_info['required_drones']
            current = len(self.target_assignments.get(target_id, []))

            if current < required and target_info['priority_score'] > 70:
                # Yüksek öncelikli hedef, yetersiz drone
                print(
                    f"[COORDINATOR] ⚠️  Hedef {target_id} ({target_info['type']}) için destek gerekli: {current}/{required} drone")

    def _assign_idle_drones(self, available_targets):
        """
        Serbest drone'ları hedeflere ata - AKILLI KAMİKAZE STRATEJİSİ

        PHASE 1: FOCUS-TO-KILL
        - Yüksek öncelikli hedefleri TAM doldur
        
        PHASE 2: OPPORTUNISTIC KILL
        - Kalanlarla küçük hedefleri avla (örn: piyade)
        
        PHASE 3: OVERFLOW
        - Hala kalan varsa en yüksek önceliğe destek at
        """
        # Serbest drone'ları bul
        idle_drones = [
            drone_id for drone_id, state in self.drone_states.items()
            if state['status'] == 'idle' and not self.env.drones[drone_id]['destroyed']
        ]

        if not idle_drones or not available_targets:
            return

        print(f"[COORDINATOR] 🔍 {len(idle_drones)} serbest drone, {len(available_targets)} hedef")

        # --- PHASE 1: FOCUS-TO-KILL ---
        for target_info in available_targets:
            if not idle_drones: break

            target_id = target_info['id']
            required = target_info['required_drones']
            current_assigned = len(self.target_assignments.get(target_id, []))
            needed = required - current_assigned

            if needed <= 0: continue

            if len(idle_drones) >= needed:
                self._assign_batch(target_info, needed, idle_drones)
                print(f"[COORDINATOR] ✅ (P1) Hedef {target_id} ({target_info['type']}) TAM ATANDI: {needed} drone")

        # --- PHASE 2: OPPORTUNISTIC KILL ---
        if idle_drones:
            for target_info in available_targets:
                if not idle_drones: break

                target_id = target_info['id']
                required = target_info['required_drones']
                current_assigned = len(self.target_assignments.get(target_id, []))
                needed = required - current_assigned

                if needed <= 0: continue

                if len(idle_drones) >= needed:
                    self._assign_batch(target_info, needed, idle_drones)
                    print(f"[COORDINATOR] 🎯 (P2) Fırsat Hedefi {target_id} ({target_info['type']}) AVLANDI: {needed} drone")

        # --- PHASE 3: OVERFLOW ---
        if idle_drones:
            target_info = available_targets[0]
            # En yüksek öncelikli hedefi seçiyoruz
            count = len(idle_drones)
            self._assign_batch(target_info, count, idle_drones)
            print(f"[COORDINATOR] 💪 (P3) Destek {target_info['id']} ({target_info['type']}): +{count} drone")

    def _assign_batch(self, target_info, count, idle_source):
        """Yardımcı fonksiyon: Belirli sayıda drone'u hedefe ata"""
        target_id = target_info['id']
        target_pos = target_info['position']
        
        drone_distances = []
        for drone_id in idle_source:
             drone = self.env.drones[drone_id]
             dist = np.sqrt((drone['x'] - target_pos[0]) ** 2 + (drone['y'] - target_pos[1]) ** 2)
             drone_distances.append((drone_id, dist))
        
        drone_distances.sort(key=lambda x: x[1])
        
        for i in range(count):
            drone_id, dist = drone_distances[i]
            
            self.target_assignments[target_id].append(drone_id)
            self.drone_states[drone_id]['status'] = 'assigned'
            self.drone_states[drone_id]['target'] = target_id
            
            # Listeden çıkar
            idle_source.remove(drone_id)

            print(
                f"[COORDINATOR] 📍 Drone {drone_id} → Hedef {target_id} ({target_info['type']}, dist={dist:.0f})"
            )
            
            if len(self.target_assignments[target_id]) == target_info['required_drones']:
                 self.mission_log.append({
                    'action': 'full_assign',
                    'target_id': target_id,
                    'type': target_info['type']
                })

    def _generate_directive(self, drone_id, observation, available_targets):
        """
        Tek bir drone için stratejik direktif oluştur

        Return: Dict {
            'target_id': int,
            'priority': float,
            'role': str,  # 'attacker', 'scout', 'support'
            'coordination': bool,  # Koordineli saldırı mı?
            'teammates': List[int]  # Aynı hedefe giden diğer drone'lar
        }
        """
        directive = {
            'target_id': -1,
            'priority': 0.0,
            'role': 'scout',
            'coordination': False,
            'teammates': [],
            'should_attack': False
        }

        # Drone durumu
        if drone_id not in self.drone_states:
            return directive

        state = self.drone_states[drone_id]

        # Eğer atanmış hedef varsa
        if state['target'] is not None:
            target_id = state['target']

            # Hedef hala mevcut mu?
            target_info = next((t for t in available_targets if t['id'] == target_id), None)

            if target_info:
                # Hedef hala var, saldırıya devam
                directive['target_id'] = target_id
                directive['priority'] = target_info['priority_score']
                directive['role'] = 'attacker'
                directive['coordination'] = True
                directive['teammates'] = [d for d in self.target_assignments.get(target_id, []) if d != drone_id]

                # Saldırı mesafesinde mi?
                drone = self.env.drones[drone_id]
                target_pos = target_info['position']
                dist = np.sqrt((drone['x'] - target_pos[0]) ** 2 + (drone['y'] - target_pos[1]) ** 2)

                directive['should_attack'] = dist <= self.env.attack_range
            else:
                # Hedef yok olmuş, serbest bırak
                state['status'] = 'idle'
                state['target'] = None
        else:
            # Serbest drone - keşif görevi
            directive['role'] = 'scout'

            # Görünen hedefler arasından en önemlisini seç
            visible_targets = observation.get('visible_targets', [])
            if visible_targets:
                # En yüksek öncelikli görünen hedefi bul
                best_target = max(visible_targets, key=lambda t: t.get('importance', 0))
                directive['target_id'] = int(best_target.get('id', -1))
                directive['priority'] = float(best_target.get('importance', 0))

        return directive

    def get_coordination_reward(self, drone_id):
        """
        Koordinasyon ödülü hesapla

        Ödül faktörleri:
        - Atanmış hedefe gidiyor mu?
        - Takım arkadaşlarıyla koordineli mi?
        - Görevini tamamladı mı?
        """
        reward = 0.0

        if drone_id not in self.drone_states:
            return reward

        state = self.drone_states[drone_id]

        # Atanmış hedef varsa ve ona gidiyorsa
        if state['target'] is not None:
            reward += 2.0  # Görev odaklı olma ödülü

            # Takım arkadaşlarıyla koordinasyon
            teammates = self.target_assignments.get(state['target'], [])
            if len(teammates) > 1:
                reward += 1.0 * len(teammates)  # Takım çalışması bonusu

        return reward

    def get_mission_summary(self):
        """Görev özeti"""
        summary = {
            'active_assignments': len(self.target_assignments),
            'idle_drones': sum(1 for s in self.drone_states.values() if s['status'] == 'idle'),
            'active_drones': sum(1 for s in self.drone_states.values() if s['status'] == 'assigned'),
            'mission_log': self.mission_log[-10:]  # Son 10 kayıt
        }
        return summary