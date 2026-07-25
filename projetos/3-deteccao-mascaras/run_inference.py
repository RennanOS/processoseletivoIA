import os
from ultralytics import YOLO

N_SAMPLES = 5
CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]

def main():
    print("=" * 60)
    print("Projeto 3 — Inferência com model.tflite (Edge AI)")
    print("=" * 60)

    # Verificar se model.tflite existe
    if not os.path.exists("model.tflite"):
        print("❌ model.tflite não encontrado!")
        return
    
    print("✅ model.tflite encontrado!")

    # Carregar modelo
    try:
        model = YOLO("model.tflite", task="detect")
        print("✅ Modelo carregado!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return

    # Listar imagens de validação
    val_dir = "dataset/images/val"
    if not os.path.exists(val_dir):
        print(f"❌ Pasta {val_dir} não encontrada!")
        return
    
    all_images = sorted([f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    sample_images = all_images[:N_SAMPLES]
    
    print(f"\nRodando inferência em {len(sample_images)} amostras:\n")
    print(f"{'Imagem':<35} {'Detecções':>10}  Detalhes")
    print("-" * 70)

    total_detections = 0

    for img_name in sample_images:
        img_path = os.path.join(val_dir, img_name)
        
        try:
            # Chamada direta do modelo (sem predict)
            results = model(img_path, conf=0.25)
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                n_det = len(boxes)
                total_detections += n_det
                
                # Contar classes
                class_counts = {}
                for box in boxes:
                    class_id = int(box.cls)
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                details = ", ".join(f"{v}x {k}" for k, v in class_counts.items())
            else:
                n_det = 0
                details = "nenhuma detecção"
            
            print(f"{img_name:<35} {n_det:>10}  [{details}]")
            
        except Exception as e:
            print(f"{img_name:<35} {'ERRO':>10}  [{str(e)[:50]}]")

    print("-" * 70)
    print(f"{'TOTAL':<35} {total_detections:>10}")
    print("\n✅ Inferência concluída!")

if __name__ == "__main__":
    main()