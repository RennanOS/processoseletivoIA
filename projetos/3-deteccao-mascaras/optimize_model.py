from ultralytics import YOLO
import os
import time
import subprocess
import sys

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("⚡ OTIMIZANDO MODELO PARA TENSORFLOW LITE (EDGE AI)")
    print("=" * 60)
    
    # 1. Verificar se o model.pt existe
    if not os.path.exists("model.pt"):
        print("❌ ERRO: model.pt não encontrado!")
        print("   Execute train_model.py primeiro para gerar o modelo.")
        return
    
    print("✅ Modelo encontrado: model.pt")
    
    # 2. Carregar o modelo treinado
    print("\n📦 Carregando modelo treinado...")
    try:
        model = YOLO("model.pt")
        print("✅ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # 3. Tentar exportar para TFLite diretamente (pode funcionar em algumas versões)
    print("\n⚡ Tentando exportar para TFLite...")
    
    try:
        start_time = time.time()
        
        # Primeiro, exportar para ONNX (funciona no Windows)
        print("\n📤 Exportando para ONNX primeiro...")
        onnx_path = model.export(
            format="onnx",
            imgsz=640,
            opset=12
        )
        print(f"✅ ONNX exportado: {onnx_path}")
        
        # Depois converter ONNX para TFLite usando onnx2tf
        print("\n🔄 Convertendo ONNX para TFLite usando onnx2tf...")
        
        # Verificar se onnx2tf está instalado
        try:
            import onnx2tf
            print("✅ onnx2tf encontrado!")
        except ImportError:
            print("⚠️ onnx2tf não encontrado. Instalando...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx2tf"])
            print("✅ onnx2tf instalado!")
        
        # Executar conversão via linha de comando
        cmd = [
            "onnx2tf",
            "-i", "model.onnx",
            "-o", ".",
            "-onnx_opset", "12",
            "-cotof", "1"
        ]
        
        print(f"Executando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Conversão ONNX → TFLite concluída!")
            if os.path.exists("model_float32.tflite"):
                os.rename("model_float32.tflite", "model.tflite")
                print("✅ model.tflite gerado com sucesso!")
        else:
            print("❌ Erro na conversão:")
            print(result.stderr)
            
            # Tentar método alternativo
            print("\n🔄 Tentando método alternativo...")
            try:
                # Usar o método de exportação com diferentes parâmetros
                model.export(
                    format="tflite",
                    imgsz=640,
                    int8=False,
                    optimize=False  # Tentar sem otimização
                )
                print("✅ Exportação direta funcionou com parâmetros alternativos!")
            except Exception as e2:
                print(f"❌ Método alternativo também falhou: {e2}")
                raise
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Tempo total: {elapsed_time:.2f} segundos")
        
        # Verificar arquivos gerados
        check_generated_files()
        
    except Exception as e:
        print(f"\n❌ Erro durante a exportação: {e}")
        print("\n🔧 Soluções alternativas:")
        print("   1. Instale as dependências manualmente:")
        print("      pip install onnx onnx2tf tensorflow-cpu")
        print("   2. Tente a Solução 3 (Google Colab) mencionada no README")
        print("   3. Use WSL (Windows Subsystem for Linux)")

def check_generated_files():
    """Verifica e mostra os arquivos gerados"""
    print("\n📂 Arquivos gerados:")
    
    # Verificar TFLite
    tflite_files = ["model.tflite", "model_float32.tflite"]
    found = False
    for file in tflite_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"   ✅ {file} - {size_mb:.2f} MB")
            found = True
    
    if not found:
        print("   ⚠️ Nenhum arquivo .tflite encontrado")
    
    # Verificar ONNX
    if os.path.exists("model.onnx"):
        size_mb = os.path.getsize("model.onnx") / (1024 * 1024)
        print(f"   📄 model.onnx - {size_mb:.2f} MB")
    
    # Verificar SavedModel
    if os.path.exists("model_saved_model"):
        print(f"   📁 model_saved_model/ (pasta)")

if __name__ == "__main__":
    main()