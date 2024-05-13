import pygame
import random
pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Collision')

line_start = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

obstacles = []
for i in range(16):
    obstacle_rect = pygame.Rect(random.randint(0, 500), random.randint(0, 300), 25, 25)
    obstacles.append(obstacle_rect)

BG = (50, 50, 50)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

running = True
while running:
    screen.fill(BG)

    pos = pygame.mouse.get_pos()

    line = pygame.draw.line(screen, WHITE, line_start, pos, 5)

    for obstacle_rect in obstacles:
        if obstacle_rect.clipline((line_start, pos)):
            pygame.draw.rect(screen, RED, obstacle_rect)
        else:
            pygame.draw.rect(screen, GREEN, obstacle_rect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    pygame.display.flip()
pygame.quit()