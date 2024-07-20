import pygame
from spritesheet import SpriteSheet

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Sprite Animation')

FPS = 60
clock = pygame.time.Clock()

BG = (25, 25, 25)
BLACK = (0, 0, 0)

spite_sheet_image = pygame.image.load('doux.png').convert_alpha()
sprite_sheet = SpriteSheet(spite_sheet_image)

animation_list = []
animation_steps = [4, 6, 3, 4, 7]
last_update = pygame.time.get_ticks()
animation_cooldown = 100
image_counter = 0
frame = 0
action = 3

for steps in animation_steps:
    temp_list = []
    for _ in range(steps):
        image = sprite_sheet.get_image(image_counter, 24, 24, 10, BLACK)
        image_counter += 1
        temp_list.append(image)
    animation_list.append(temp_list)

running = True
while running:
    clock.tick(FPS)

    screen.fill(BG)

    current_time = pygame.time.get_ticks()
    if current_time - last_update > animation_cooldown:
        frame += 1
        last_update = current_time
        if frame >= len(animation_list[action]):
            frame = 0

    x = screen.get_rect().centerx - animation_list[action][frame].get_rect().centerx
    y = screen.get_rect().centery - animation_list[action][frame].get_rect().centery
    screen.blit(animation_list[action][frame], (x, y))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                action += 1
                action %= len(animation_list)
                frame = 0
            if event.key == pygame.K_s:
                action -= 1
                action %= len(animation_list)
                frame = 0

    pygame.display.update()

pygame.quit()