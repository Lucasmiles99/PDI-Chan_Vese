import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200):
    """Aplica o filtro Chan-Vese e retorna tanto o level set quanto a máscara."""
    # Pré-processamento
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normaliza a imagem
    image = image.astype(np.float32) / 255
    
    # Inicialização do level set
    height, width = image.shape
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    phi = ((y - center_y)**2 / (height/3)**2 + 
           (x - center_x)**2 / (width/2.5)**2 - 1)
    
    # Iterações do Chan-Vese
    dt = 0.5
    for iteration in range(max_iter):
        # Define regiões
        inside = phi >= 0
        outside = phi < 0
        
        # Calcula médias
        c1 = np.mean(image[inside]) if np.any(inside) else 0
        c2 = np.mean(image[outside]) if np.any(outside) else 0
        
        # Atualiza level set
        curvatura = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
        fitting = (lambda1 * (image - c1)**2 - lambda2 * (image - c2)**2)
        dphi = dt * (mu * curvatura - fitting)
        phi += dphi
        
        if np.max(np.abs(dphi)) < tol:
            break
    
    return phi, image

def main():
    # Carrega a imagem
    image = cv2.imread('galaxia.jpeg')
    if image is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Redimensiona para um tamanho gerenciável
    max_size = 800
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Aplica o filtro
    phi, gray_image = apply_chan_vese(image)
    mask = (phi >= 0).astype(np.uint8) * 255
    
    # Configuração da figura
    plt.figure(figsize=(15, 5))
    
    # Imagem original
    plt.subplot(131)
    plt.title('Imagem Original')
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    
    # Imagem com contorno
    plt.subplot(132)
    plt.title('Contorno Chan-Vese')
    plt.imshow(gray_image, cmap='gray')
    plt.contour(phi, [0], colors='white', linewidths=2)
    plt.axis('off')
    
    # Máscara binária
    plt.subplot(133)
    plt.title('Máscara Binária')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()