import pygame
import random
pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Collision')


rect1 = pygame.Rect(0, 0, 25, 25)
obstacles = []
for i in range(16):
    obstacle_rect = pygame.Rect(random.randint(0, 500), random.randint(0, 300), 25, 25)
    obstacles.append(obstacle_rect)

BG = (50, 50, 50)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

pygame.mouse.set_visible(False)


running = True
while running:
    screen.fill(BG)

    rect1.center = pygame.mouse.get_pos()
    col = GREEN

    # for obstacle_rect in obstacles:
    #     if rect1.colliderect(obstacle_rect):
    #         col = RED

    if (collision_id := rect1.collidelist(obstacles)) >= 0:
        print(collision_id)
        col = RED

    pygame.draw.rect(screen, col, rect1)
    for obstacle_rect in obstacles:
        pygame.draw.rect(screen, BLUE, obstacle_rect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # pygame.display.update()
    pygame.display.flip()
pygame.quit()