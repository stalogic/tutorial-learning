import pygame


pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_CENTER = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pixel Perfect Collision")

# 定义颜色
BG = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

pygame.mouse.set_visible(False)

player = pygame.image.load("amber.png").convert_alpha()
player_width = 100
player_height = player.get_height() * player_width / player.get_width()
player = pygame.transform.scale(player, (player_width, player_height))
player_rect = player.get_rect()

player_mask = pygame.mask.from_surface(player)
mask_img = player_mask.to_surface()

# 创建子弹
bullet = pygame.Surface((10, 10))
bullet.fill(GREEN)
bullet_mask = pygame.mask.from_surface(bullet)

# 移动player
player_rect.topleft = SCREEN_CENTER

running = True
while running:

    pos = pygame.mouse.get_pos()

    screen.fill(BG)
    screen.blit(player, player_rect)
    screen.blit(mask_img, (0, 0))
    pygame.draw.rect(screen, RED, player_rect, 1)
    
    # 碰撞检测
    # 获取两个图形的Mask，然后调用overlap方法，a_mask.overlap(b_mask, offset)
    # offset是两个图形的相对位置 a_rect - b_rect
    if player_mask.overlap(bullet_mask, (pos[0] - player_rect.x, pos[1] - player_rect.y)):
        col = RED
    else:
        col = GREEN

    bullet.fill(col)
    screen.blit(bullet, pos)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
pygame.quit()
