from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

def smote_parsial():
    return SMOTE(sampling_strategy=target_ros, k_neighbors=K, random_state=RANDOM_STATE)

# a) ENN bawaan: kind_sel="all" pada semua kelas
_ = jalankan("SMOTE-ENN", SMOTEENN(smote=smote_parsial(), random_state=RANDOM_STATE))

# b) ENN lebih ringan: kind_sel="mode"
enn_mode = EditedNearestNeighbours(sampling_strategy="all", kind_sel="mode")
_ = jalankan("SMOTE-ENN mode", SMOTEENN(smote=smote_parsial(), enn=enn_mode,
                                        random_state=RANDOM_STATE))

for nama in ("SMOTE-ENN", "SMOTE-ENN mode"):
    dibuang = {k: hasil["SMOTE parsial"][k] - hasil[nama][k] for k in nama_kelas}
    print(nama, "| total dibuang:", sum(dibuang.values()), "|", dibuang)
gambar("SMOTE-ENN")
