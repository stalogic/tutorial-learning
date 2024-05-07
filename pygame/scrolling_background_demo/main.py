import pygame
import math
import time

pygame.init()
clock = pygame.time.Clock()
FPS = 60

SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 600

# 创建窗口
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("无限滚动背景")

bg = pygame.image.load("bg.png").convert()
bg_width = bg.get_width()
bg_rect = bg.get_rect()

# 背景图重复
scroll = 0
tiles = math.ceil(SCREEN_WIDTH / bg_width) + 1


font = pygame.font.SysFont("arial", 20)

def draw_text(screen, text, x, y, color=(0, 0, 0)):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


last = time.time()
# 主循环
running = True
while running:

    for i in range(-1, tiles):
        screen.blit(bg, (i * bg_width + scroll, 0))
        bg_rect.x = i * bg_width + scroll
        pygame.draw.rect(screen, (255, 0, 0), bg_rect, 1)

    if pygame.key.get_pressed()[pygame.K_a]:
        scroll -= 5
    elif pygame.key.get_pressed()[pygame.K_d]:
        scroll += 5
    if abs(scroll) > bg_width:
        scroll = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    gap = time.time() - last
    last = time.time()
    fps_info = f"Frame Generate Time: {gap:.3f}s, FPS: {1/gap:.3f}, Sys FPS: {clock.get_fps():.3f}, Scroll: {scroll}"
    draw_text(screen, fps_info, 100, 100)
    pygame.display.update()
    clock.tick(FPS)

pygame.quit()
