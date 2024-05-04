import pygame

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 640

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Demo")

player = pygame.Rect(300, 250, 50, 50)

running = True
while running:

    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, (255, 0, 0), player)

    key = pygame.key.get_pressed()

    if key[pygame.K_a]:
        player.move_ip(-1, 0)
    elif key[pygame.K_d]:
        player.move_ip(1, 0)
    elif key[pygame.K_w]:
        player.move_ip(0, -1)
    elif key[pygame.K_s]:
        player.move_ip(0, 1)
    
    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                running = False

    pygame.display.update()

pygame.quit()

