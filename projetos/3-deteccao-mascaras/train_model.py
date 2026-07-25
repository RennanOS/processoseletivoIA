from ultralytics import YOLO
import os
import shutil

def main():
    # 1. Carregar o modelo pré-treinado YOLO11n
    print("📦 Carregando modelo YOLO11n pré-treinado...")
    model = YOLO("yolo11n.pt")
    
    # 2. Verificar se o dataset existe
    dataset_yaml = "dataset/data.yaml"
    if not os.path.exists(dataset_yaml):
        print(f"❌ ERRO: Arquivo {dataset_yaml} não encontrado!")
        return
    
    print("✅ Dataset encontrado!")
    
    # 3. Fine-tuning do modelo
    print("🚀 Iniciando fine-tuning...")
    print("⏱️ Isso pode levar alguns minutos...")
    
    results = model.train(
        data=dataset_yaml,          # Caminho para o arquivo data.yaml
        epochs=20,                  # Número de épocas (15-30 é razoável)
        imgsz=640,                  # Tamanho das imagens
        batch=8,                    # Tamanho do batch (ajuste se tiver pouca RAM)
        device="cpu",               # Forçar uso de CPU
        workers=4,                  # Número de workers para loading de dados
        patience=10,                # Early stopping se não melhorar
        project="runs/detect",      # Onde salvar os resultados
        name="train",               # Nome da execução
        exist_ok=True,              # Sobrescrever se existir
        verbose=True                # Mostrar progresso
    )
    
    print("✅ Fine-tuning concluído!")
    
    # 4. Copiar os melhores pesos para a raiz do projeto
    best_weights_path = "runs/detect/runs/detect/train/weights/best.pt"
    destination_path = "model.pt"
    
    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, destination_path)
        print(f"✅ Modelo salvo em: {destination_path}")
        print(f"📊 Métricas finais: mAP50 = {results.maps[0]:.3f}")
    else:
        print(f"❌ ERRO: Arquivo {best_weights_path} não encontrado!")
    
    print("🎯 Fine-tuning finalizado com sucesso!")

if __name__ == "__main__":
    main()