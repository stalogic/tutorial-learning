import pygame

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 490

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3D BG Effect")

clock = pygame.time.Clock()
FPS = 60

scroll = 0

ground = pygame.image.load("ground.png").convert_alpha()
ground_width = ground.get_width()
ground_height = ground.get_height()

bg_images = []
for i in range(1, 6):
    image = pygame.image.load(f"plx-{i}.png").convert_alpha()
    bg_images.append(image)

bg_width = bg_images[0].get_width()

# 不同背景图层具有不同移动速度，形成3D效果， 背景越远速度越慢
def draw_bg():
    for x in range(5):
        speed = 1
        for image in bg_images:
            screen.blit(image, (x*bg_width - scroll * speed, 0))
            speed += 0.2

def draw_ground():
    for x in range(15):
        screen.blit(ground, (x*ground_width - scroll * 1.8, SCREEN_HEIGHT - ground_height))

running = True
while running:
    clock.tick(FPS)

    draw_bg()
    draw_ground()

    key = pygame.key.get_pressed()
    if key[pygame.K_a] and scroll > 0:
        scroll -= 5
    if key[pygame.K_d] and scroll < 1500:
        scroll += 5

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()

pygame.quit()