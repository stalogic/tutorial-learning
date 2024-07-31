import pygame

pygame.init()

FPS = 60
clock = pygame.time.Clock()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Spritesheets')

sprite_sheet_image = pygame.image.load('doux.png').convert_alpha()

BG = (50, 50, 50)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

def get_image(sheet, frame, width, height, scale=1, color_key=BLACK):
    image = pygame.Surface((width, height)).convert_alpha()
    image.blit(sheet, (0, 0), (frame * width, 0, width, height))
    image = pygame.transform.scale(image, (width * scale, height * scale))
    image.set_colorkey(color_key)
    return image


frame_0 = get_image(sprite_sheet_image, 0, 24, 24, 10, BLACK)
frame_1 = get_image(sprite_sheet_image, 1, 24, 24, 10, BLACK)
frame_2 = get_image(sprite_sheet_image, 2, 24, 24, 10, BLACK)
frame_3 = get_image(sprite_sheet_image, 3, 24, 24, 10, BLACK)
frame_4 = get_image(sprite_sheet_image, 4, 24, 24, 10, BLACK)

frames = [frame_0, frame_1, frame_2, frame_3, frame_4]

index = 0

running = True
while running:
    clock.tick(FPS)

    screen.fill(BG)
    screen.blit(frame_0, (0, 0))
    screen.blit(frame_1, (300, 0))
    screen.blit(frame_2, (600, 0))
    screen.blit(frame_3, (0, 300))
    screen.blit(frame_4, (300, 300))

    screen.blit(frames[index // 15], (600, 300))
    index = (index + 1) % (15 * len(frames))


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()