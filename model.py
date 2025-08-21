import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np
import time

# Создаем директории для сохранения
os.makedirs("conditional_diffusion", exist_ok=True)
os.makedirs("generated_digits", exist_ok=True)

# Конфигурация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 128
image_size = 28
channels = 1
num_classes = 10
epochs = 20
T = 1000
lr = 1e-3

# Расписание диффузии
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1. / alphas)

# Перенос тензоров на устройство
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)

# Позиционное кодирование времени
def timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# Блок U-Net с условием на класс
class ConditionalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, class_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.class_mlp = nn.Linear(class_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t_emb, class_emb):
        h = self.conv1(x)
        h = self.norm(h)
        h = self.act(h)
        
        # Добавляем информацию о времени и классе
        t_emb = self.time_mlp(self.act(t_emb))
        c_emb = self.class_mlp(self.act(class_emb))
        h = h + t_emb[..., None, None] + c_emb[..., None, None]
        
        h = self.conv2(h)
        h = self.norm(h)
        h = self.act(h)
        return h

# Down-блок с условием
class ConditionalDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, class_emb_dim):
        super().__init__()
        self.block = ConditionalBlock(in_ch, out_ch, time_emb_dim, class_emb_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb, class_emb):
        x = self.block(x, t_emb, class_emb)
        skip = x
        x = self.down(x)
        return x, skip

# Up-блок с условием
class ConditionalUpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, class_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = ConditionalBlock(in_ch + skip_ch, out_ch, time_emb_dim, class_emb_dim)

    def forward(self, x, skip, t_emb, class_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb, class_emb)
        return x

# U-Net с поддержкой условий
class ConditionalUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, time_emb_dim=128, class_emb_dim=64):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Вложение для классов
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(class_emb_dim, class_emb_dim),
            nn.SiLU(),
            nn.Linear(class_emb_dim, class_emb_dim)
        )
        
        self.init_conv = nn.Conv2d(in_ch, 64, 3, padding=1)
        
        # Downsampling
        self.down1 = ConditionalDownBlock(64, 128, time_emb_dim, class_emb_dim)
        self.down2 = ConditionalDownBlock(128, 256, time_emb_dim, class_emb_dim)
        
        # Bottleneck
        self.bottleneck = ConditionalBlock(256, 512, time_emb_dim, class_emb_dim)
        
        # Upsampling
        self.up1 = ConditionalUpBlock(512, 256, 256, time_emb_dim, class_emb_dim)
        self.up2 = ConditionalUpBlock(256, 128, 128, time_emb_dim, class_emb_dim)
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, x, t, y):
        # Временные embedding
        t_emb = timestep_embedding(t, 128)
        t_emb = self.time_emb(t_emb)
        
        # Классовые embedding
        class_emb = self.class_emb(y)
        class_emb = self.class_mlp(class_emb)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsample
        x, skip1 = self.down1(x, t_emb, class_emb)
        x, skip2 = self.down2(x, t_emb, class_emb)
        
        # Bottleneck
        x = self.bottleneck(x, t_emb, class_emb)
        
        # Upsample
        x = self.up1(x, skip2, t_emb, class_emb)
        x = self.up2(x, skip1, t_emb, class_emb)
        
        # Final convolution
        x = self.final_conv(x)
        return x

# Направленная диффузионная модель
class ConditionalDiffusionModel:
    def __init__(self):
        self.model = ConditionalUNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.T = T
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt, noise

    def train_step(self, x0, y):
        self.optimizer.zero_grad()
        t = torch.randint(0, self.T, (x0.shape[0],), device=device)
        xt, noise = self.forward_diffusion(x0, t)
        noise_pred = self.model(xt, t, y)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, y, n=None):
        if n is None:
            n = len(y)
        self.model.eval()
        x = torch.randn((n, channels, image_size, image_size), device=device)
        
        for t in range(self.T-1, -1, -1):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            noise_pred = self.model(x, t_batch, y)
            
            # Коэффициенты для обратного процесса
            alpha_t = alphas[t]
            beta_t = betas[t]
            sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
            
            # Вычисление среднего
            if t > 0:
                mean = sqrt_recip_alphas[t] * (x - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)
            else:
                mean = x
                
            # Добавление шума
            if t > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * z
            else:
                x = mean
                
        self.model.train()
        return x

    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filename}")

def main():
    # Загрузка данных
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Инициализация модели
    diffusion = ConditionalDiffusionModel()
    model_filename = "simple-conditiondiff.pt"
    
    # Проверяем, существует ли сохраненная модель
    if os.path.exists(model_filename):
        print("Найдена сохраненная модель. Загружаем...")
        diffusion.load_model(model_filename)
    else:
        # Обучение
        print("Начинаем обучение модели...")
        for epoch in range(epochs):
            total_loss = 0
            for i, (images, labels) in enumerate(dataloader):
                x0 = images.to(device)
                y = labels.to(device)
                loss = diffusion.train_step(x0, y)
                total_loss += loss
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss:.4f}")
            
            # Генерация примеров для каждого класса
            print("Генерация примеров...")
            all_samples = []
            for class_idx in range(10):
                y = torch.tensor([class_idx] * 4, device=device)
                samples = diffusion.sample(y)
                all_samples.append(samples)
            
            # Сохранение результатов
            all_samples = torch.cat(all_samples, dim=0)
            save_image(all_samples, f"conditional_diffusion/samples_epoch_{epoch+1}.png", 
                      nrow=10, normalize=True)
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")
        
        # Сохраняем модель после обучения
        diffusion.save_model(model_filename)
    
    # Интерактивный режим генерации
    print("Интерактивный режим генерации. Введите цифру от 0 до 9 для генерации или 'q' для выхода.")
    while True: 
        user_input = input("Введите цифру: ")
        if user_input.lower() == 'q':
            print("Выход из программы.")
            break
        
        try:
            digit = int(user_input)
            if digit < 0 or digit > 9:
                print("Пожалуйста, введите цифру от 0 до 9.")
                continue
            
            # Генерация изображения
            y = torch.tensor([digit], device=device)
            generated_image = diffusion.sample(y, n=1)
            
            # Сохранение и отображение
            timestamp = int(time.time())
            filename = f"generated_digits/digit_{digit}_{timestamp}.png"
            save_image(generated_image, filename, normalize=True)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(generated_image[0].permute(1, 2, 0).cpu().squeeze(), cmap='gray')
            plt.title(f"Сгенерированная цифра: {digit}")
            plt.axis('off')
            plt.savefig(f"generated_digits/digit_{digit}_{timestamp}_plot.png")
            plt.show()
            
            print(f"Изображение сохранено как {filename}")
            
        except ValueError:
            print("Пожалуйста, введите корректную цифру или 'q' для выхода.")

if __name__ == "__main__":
    main()
