import pandas as pd
import pulp as pl
import os

# =====================================================
# 1. EXCEL VERİSİNİ OKUMA VE HAZIRLIK
# =====================================================
excel_file = "VeriSeti.xlsx"

if not os.path.exists(excel_file):
    print(f"HATA: '{excel_file}' bulunamadı. Lütfen dosyayı çalışma ortamına yükleyin.")
else:
    print("E.xlsx dosyası başarıyla bulundu. Model kusursuz modda başlatılıyor...\n")

    df_sabah_sef = pd.read_excel(excel_file, sheet_name="sabah seferleri")
    df_aksam_sef = pd.read_excel(excel_file, sheet_name="aksam seferleri")
    df_sabah_sof = pd.read_excel(excel_file, sheet_name="sabah soforleri")
    df_aksam_sof = pd.read_excel(excel_file, sheet_name="aksam soforleri")
    df_arac = pd.read_excel(excel_file, sheet_name="otobusler")
    df_param = pd.read_excel(excel_file, sheet_name="parametreler")

    # =====================================================
    # 2. PARAMETRE VE SAAT HESAPLAMALARI
    # =====================================================
    param_dict = dict(zip(df_param.iloc[:, 0], df_param.iloc[:, 1]))

    MIN_MOLA = float(param_dict.get("MinMola", 60))
    TOPLAM_MESAI = float(param_dict.get("toplam_mesai_suresi", 480))
    FAZLA_MESAI_CARPAN = float(param_dict.get("fazla_mesai_ucreti", 1.5))
    AYLIK_MAAS = float(param_dict.get("aylık_toplam_maas", 4543000))

    I_sabah = df_sabah_sof['sabah_Sofor_ID'].tolist()
    I_aksam = df_aksam_sof['aksam_Sofor_ID'].tolist()
    I_all = I_sabah + I_aksam
    V = df_arac.iloc[:, 0].tolist()

    C_gunluk_sabit = AYLIK_MAAS / 30.0
    gunluk_kisi_basi = C_gunluk_sabit / len(I_all)
    saatlik_ucret = gunluk_kisi_basi / 8.0
    OC_dakika = (saatlik_ucret / 60.0) * FAZLA_MESAI_CARPAN

    def time_to_mins(t_obj):
        if isinstance(t_obj, str):
            h, m = map(int, str(t_obj).split(':')[:2])
            return h * 60 + m
        else:
            return t_obj.hour * 60 + t_obj.minute

    S_sabah = df_sabah_sef['sabah_Sefer_ID'].tolist()
    S_aksam = df_aksam_sef['aksam_Sefer_ID'].tolist()
    S_all = S_sabah + S_aksam

    sure, zorluk, baslangic, bitis, vardiya_tipi = {}, {}, {}, {}, {}
    saat_asimi_maliyet, saat_asimi_sure = {}, {}
    endpoints = set()

    # Sabah Seferleri İşleniyor
    for idx, row in df_sabah_sef.iterrows():
        s = row['sabah_Sefer_ID']
        sure[s] = float(row['sabah_Sure_dk'])
        zorluk[s] = float(row['sabah_Zorluk_Puani'])
        bas = time_to_mins(row['sabah_Baslangic'])
        bit = time_to_mins(row['sabah_Bitis'])
        if bit < bas: bit += 1440

        baslangic[s], bitis[s], vardiya_tipi[s] = bas, bit, 'Sabah'
        ot_dk = max(0, bit - 840)
        saat_asimi_sure[s] = ot_dk
        saat_asimi_maliyet[s] = ot_dk * OC_dakika
        endpoints.update([bas, bit])

    # Akşam Seferleri İşleniyor
    for idx, row in df_aksam_sef.iterrows():
        s = row['aksam_Sefer_ID']
        sure[s] = float(row['aksam_Sure_dk'])
        zorluk[s] = float(row['aksam_Zorluk_Puani'])
        bas = time_to_mins(row['aksam_Baslangic'])
        bit = time_to_mins(row['aksam_Bitis'])
        if bit < bas: bit += 1440

        baslangic[s], bitis[s], vardiya_tipi[s] = bas, bit, 'Aksam'
        ot_dk = max(0, bit - 1320)
        saat_asimi_sure[s] = ot_dk
        saat_asimi_maliyet[s] = ot_dk * OC_dakika
        endpoints.update([bas, bit])

    # --- KAPASİTE ANALİZÖRÜ (Siz sefer sildiğiniz için bu kez TIKANMAYACAK) ---
    events = []
    for s in S_all:
        events.append((baslangic[s], 1))
        events.append((bitis[s], -1))
    events.sort(key=lambda x: (x[0], x[1]))

    max_arac_ihtiyaci = 0
    curr = 0
    for e in events:
        curr += e[1]
        if curr > max_arac_ihtiyaci:
            max_arac_ihtiyaci = curr

    print("--------------------------------------------------")
    print(f"FİLODAKİ TOPLAM ARAÇ SAYISI: {len(V)}")
    print(f"GÜNCEL VERİDE AYNI ANDA ÇAKIŞAN MAKSİMUM SEFER: {max_arac_ihtiyaci}")
    print("--------------------------------------------------")

    if max_arac_ihtiyaci > len(V):
        raise ValueError(f"HATA: Hala eksik araç var. Max ihtiyaç {max_arac_ihtiyaci}, Filo: {len(V)}")
    else:
        print("Harika! Fiziksel kapasite sorunu çözülmüş. Optimizasyon devam ediyor...")

    sorted_endpoints = sorted(list(endpoints))
    time_points = [(sorted_endpoints[k] + sorted_endpoints[k+1]) / 2.0 for k in range(len(sorted_endpoints)-1)]

    # =====================================================
    # 3. ZİMMETLİ ARAÇ (BODY) ÖN ATAMASI
    # =====================================================
    zimmet_sabah = {}
    for idx, i in enumerate(I_sabah):
        zimmet_sabah[i] = V[idx % len(V)]

    # =====================================================
    # 4. MATEMATİKSEL MODEL VE KISITLAR
    # =====================================================
    model = pl.LpProblem("TULAS_Kusursuz_Body_Optimizasyonu", pl.LpMinimize)

    valid_pairs = [(i, s) for i in I_sabah for s in S_sabah] + \
                  [(j, s) for j in I_aksam for s in S_aksam]
    x = pl.LpVariable.dicts("Assign", valid_pairs, cat='Binary')
    z_aksam = pl.LpVariable.dicts("AksamAracZimmeti", ((j, v) for j in I_aksam for v in V), cat='Binary')

    SureAsimi = pl.LpVariable.dicts("SureAsimi", I_all, lowBound=0, cat='Continuous')

    D = pl.LpVariable.dicts("Difficulty", I_all, lowBound=0, cat='Continuous')
    Dmax_sabah = pl.LpVariable("Dmax_sabah", lowBound=0, cat='Continuous')
    Dmin_sabah = pl.LpVariable("Dmin_sabah", lowBound=0, cat='Continuous')
    Dmax_aksam = pl.LpVariable("Dmax_aksam", lowBound=0, cat='Continuous')
    Dmin_aksam = pl.LpVariable("Dmin_aksam", lowBound=0, cat='Continuous')

    # --- KISIT 1: KESİN ATAMA ---
    for s in S_sabah:
        model += pl.lpSum(x[i, s] for i in I_sabah) == 1
    for s in S_aksam:
        model += pl.lpSum(x[j, s] for j in I_aksam) == 1

    # --- KISIT 2: AKŞAM ARAÇ ZİMMETİ ---
    for j in I_aksam:
        model += pl.lpSum(z_aksam[j, v] for v in V) == 1
    for v in V:
        model += pl.lpSum(z_aksam[j, v] for j in I_aksam) <= 1

    # --- KISIT 3: CLIQUE İLE KUSURSUZ ÇAKIŞMA ENGELLEYİCİ ---
    for t in time_points:
        aktif_sabah = [s for s in S_sabah if baslangic[s] <= t < bitis[s]]
        aktif_aksam = [s for s in S_aksam if baslangic[s] <= t < bitis[s]]

        # A) Şoför Çakışmaları
        if len(aktif_sabah) > 1:
            for i in I_sabah:
                model += pl.lpSum(x[i, s] for s in aktif_sabah) <= 1

        if len(aktif_aksam) > 1:
            for j in I_aksam:
                model += pl.lpSum(x[j, s] for s in aktif_aksam) <= 1

        # B) Body (Araç) Çakışmaları
        if len(aktif_sabah) > 0 and len(aktif_aksam) > 0:
            for i in I_sabah:
                v_i = zimmet_sabah[i]
                for j in I_aksam:
                    model += pl.lpSum(x[i, s] for s in aktif_sabah) + pl.lpSum(x[j, r] for r in aktif_aksam) + z_aksam[j, v_i] <= 2

    # --- KISIT 4: ESNEK AKTİF SÜRÜŞ ---
    MAX_AKTIF = TOPLAM_MESAI - MIN_MOLA
    for i in I_sabah:
        model += pl.lpSum(sure[s] * x[i, s] for s in S_sabah) <= MAX_AKTIF + SureAsimi[i]
    for j in I_aksam:
        model += pl.lpSum(sure[s] * x[j, s] for s in S_aksam) <= MAX_AKTIF + SureAsimi[j]

    # --- KISIT 5: VARDİYA BAZLI ADALET HESAPLAMASI ---
    for i in I_sabah:
        model += D[i] == pl.lpSum(zorluk[s] * x[i, s] for s in S_sabah)
        model += Dmax_sabah >= D[i]
        model += Dmin_sabah <= D[i]
    for j in I_aksam:
        model += D[j] == pl.lpSum(zorluk[s] * x[j, s] for s in S_aksam)
        model += Dmax_aksam >= D[j]
        model += Dmin_aksam <= D[j]

    # --- AMAÇ FONKSİYONLARI ---
    Z1_Maliyet = pl.lpSum(saat_asimi_maliyet[s] * x[i, s] for (i, s) in valid_pairs) + pl.lpSum(SureAsimi[i] * OC_dakika for i in I_all)
    Z2_Adalet = (Dmax_sabah - Dmin_sabah) + (Dmax_aksam - Dmin_aksam)

    model += 1.0 * Z1_Maliyet + 10000.0 * Z2_Adalet

    # =====================================================
    # 5. ÇÖZÜM VE EXCEL RAPORLAMA
    # =====================================================
    print("Mükemmel Body Algoritması çözülüyor... Lütfen bekleyin (Maks 5 DK).")
    model.solve(pl.PULP_CBC_CMD(timeLimit=300, gapRel=0.02, msg=1))

    if pl.LpStatus[model.status] == 'Optimal' or (pl.LpStatus[model.status] == 'Not Solved' and pl.value(model.objective) is not None):
        print("\n%100 KUSURSUZ ATAMA BULUNDU! Excel dosyası hazırlanıyor...")

        detayli_atamalar, performans_sabah, performans_aksam = [], [], []
        gerceklesen_saat_asimi_dk, gerceklesen_saat_maliyet = 0, 0

        for s in S_all:
            atanan_sofor = None
            atanan_arac = None

            if vardiya_tipi[s] == 'Sabah':
                for i in I_sabah:
                    if x[i, s].varValue is not None and x[i, s].varValue > 0.5:
                        atanan_sofor = i
                        atanan_arac = zimmet_sabah[i]
                        gerceklesen_saat_asimi_dk += saat_asimi_sure[s]
                        gerceklesen_saat_maliyet += saat_asimi_maliyet[s]
                        break
            else:
                for j in I_aksam:
                    if x[j, s].varValue is not None and x[j, s].varValue > 0.5:
                        atanan_sofor = j
                        for v in V:
                            if z_aksam[j, v].varValue is not None and z_aksam[j, v].varValue > 0.5:
                                atanan_arac = v
                                break
                        gerceklesen_saat_asimi_dk += saat_asimi_sure[s]
                        gerceklesen_saat_maliyet += saat_asimi_maliyet[s]
                        break

            detayli_atamalar.append([
                s, vardiya_tipi[s],
                f"{baslangic[s]//60:02d}:{baslangic[s]%60:02d}",
                f"{bitis[s]//60:02d}:{bitis[s]%60:02d}",
                round(saat_asimi_maliyet[s], 2), atanan_sofor, atanan_arac
            ])

        sabah_sureler, aksam_sureler = [], []
        sabah_zorluklar, aksam_zorluklar = [], []
        tum_sureler = []
        gerceklesen_sure_asimi_dk = 0

        for i in I_all:
            if i in I_sabah:
                zimmetli_arac = zimmet_sabah[i]
            else:
                zimmetli_arac = "Atanmadı"
                for v in V:
                    if z_aksam[i, v].varValue is not None and z_aksam[i, v].varValue > 0.5:
                        zimmetli_arac = v
                        break

            grup_seferleri = S_sabah if i in I_sabah else S_aksam
            s_sayisi = sum(1 for s in grup_seferleri if x[i, s].varValue is not None and x[i, s].varValue > 0.5)
            s_sure = sum(sure[s] for s in grup_seferleri if x[i, s].varValue is not None and x[i, s].varValue > 0.5)
            s_zorluk = round(D[i].varValue or 0, 2)

            asım_dk = round(SureAsimi[i].varValue or 0, 2)
            gerceklesen_sure_asimi_dk += asım_dk

            tum_sureler.append(s_sure)

            if i in I_sabah:
                performans_sabah.append([i, zimmetli_arac, s_sayisi, s_sure, asım_dk, s_zorluk])
                if s_sayisi > 0:
                    sabah_sureler.append(s_sure)
                    sabah_zorluklar.append(s_zorluk)
            else:
                performans_aksam.append([i, zimmetli_arac, s_sayisi, s_sure, asım_dk, s_zorluk])
                if s_sayisi > 0:
                    aksam_sureler.append(s_sure)
                    aksam_zorluklar.append(s_zorluk)

        gerceklesen_sure_maliyeti = gerceklesen_sure_asimi_dk * OC_dakika
        toplam_mesai_maliyeti = gerceklesen_saat_maliyet + gerceklesen_sure_maliyeti
        toplam_maliyet_Z1 = C_gunluk_sabit + toplam_mesai_maliyeti
        toplam_ot_saati = (gerceklesen_saat_asimi_dk + gerceklesen_sure_asimi_dk) / 60.0

        max_z_sabah = max(sabah_zorluklar) if sabah_zorluklar else 0
        min_z_sabah = min(sabah_zorluklar) if sabah_zorluklar else 0
        max_z_aksam = max(aksam_zorluklar) if aksam_zorluklar else 0
        min_z_aksam = min(aksam_zorluklar) if aksam_zorluklar else 0

        summary_data = [
            ["Sabah Vardiyası Maksimum Çalışma Süresi (Dk)", max(sabah_sureler) if sabah_sureler else 0],
            ["Sabah Vardiyası Minimum Çalışma Süresi (Dk)", min(sabah_sureler) if sabah_sureler else 0],
            ["Akşam Vardiyası Maksimum Çalışma Süresi (Dk)", max(aksam_sureler) if aksam_sureler else 0],
            ["Akşam Vardiyası Minimum Çalışma Süresi (Dk)", min(aksam_sureler) if aksam_sureler else 0],
            ["Tüm Şoförler Ortalama Çalışma Süresi (Dk)", round(sum(tum_sureler)/len(tum_sureler), 2) if tum_sureler else 0],
            ["------------------------------------", "-------"],
            ["Sabah Grubunda Maksimum Zorluk", max_z_sabah],
            ["Sabah Grubunda Minimum Zorluk", min_z_sabah],
            ["Sabah Grubu Zorluk Farkı (Adalet)", round(max_z_sabah - min_z_sabah, 2)],
            ["------------------------------------", "-------"],
            ["Akşam Grubunda Maksimum Zorluk", max_z_aksam],
            ["Akşam Grubunda Minimum Zorluk", min_z_aksam],
            ["Akşam Grubu Zorluk Farkı (Adalet)", round(max_z_aksam - min_z_aksam, 2)],
            ["------------------------------------", "-------"],
            ["Saat Aşımından Doğan Fazla Mesai (Saat)", round(gerceklesen_saat_asimi_dk / 60, 2)],
            ["Vardiya Sürüş Aşımından Doğan Fazla Mesai (Saat)", round(gerceklesen_sure_asimi_dk / 60, 2)],
            ["Toplam Fazla Mesai Süresi (Saat)", round(toplam_ot_saati, 2)],
            ["Toplam Fazla Mesai Maliyeti (TL)", round(toplam_mesai_maliyeti, 2)],
            ["Toplam Operasyonel Maliyet Z1 (TL)", round(toplam_maliyet_Z1, 2)]
        ]

        output_file = "VeriSeti_Ciktisi.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            pd.DataFrame(summary_data, columns=["Performans Metriği", "Değer"]).to_excel(writer, sheet_name="Yönetici Özeti", index=False)
            pd.DataFrame(detayli_atamalar, columns=["Sefer", "Vardiya", "Baslangic", "Bitis", "Mesai Maliyeti (TL)", "Sofor", "Arac"]).to_excel(writer, sheet_name="Detaylı Atamalar", index=False)
            pd.DataFrame(performans_sabah, columns=["Sabah Şoförü", "Zimmetli Araç", "Sefer Sayısı", "Aktif Sürüş (Dk)", "Süre Aşımı Mesaisi (Dk)", "Toplam Zorluk"]).to_excel(writer, sheet_name="Sabah Performans", index=False)
            pd.DataFrame(performans_aksam, columns=["Akşam Şoförü", "Zimmetli Araç", "Sefer Sayısı", "Aktif Sürüş (Dk)", "Süre Aşımı Mesaisi (Dk)", "Toplam Zorluk"]).to_excel(writer, sheet_name="Akşam Performans", index=False)

        print(f"\nİşlem Başarılı! Kusursuz Excel dosyası kaydedildi: {output_file}")

        try:
            from google.colab import files
            files.download(output_file)
        except:
            pass
    else:
        print("\nÇÖZÜM BULUNAMADI! Lütfen verilerinizi tekrar kontrol edin.")