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

# 创建enemy类
class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, width = 100):
        pygame.sprite.Sprite.__init__(self)
        image = pygame.image.load("amber.png").convert_alpha()
        height = image.get_height() * width / image.get_width()
        self.image = pygame.transform.scale(image, (width, height))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.mask = pygame.mask.from_surface(self.image)

# 创建子弹类
class Bullet(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, color):
        pos = pygame.mouse.get_pos()
        self.rect.center = pos
        self.image.fill(color)



pygame.mouse.set_visible(False)

# 创建enemy和bullet的实例
enemy1 = Enemy(200, 200)
enemy2 = Enemy(600, 200)
enemy3 = Enemy(400, 200)
enemy4 = Enemy(400, 400)
bullet = Bullet()

# 创建enemy和bullet的Groups
enemy_group = pygame.sprite.Group()
bullet_group = pygame.sprite.Group()

# 添加实力到Group
enemy_group.add(enemy1, enemy2, enemy3, enemy4)
bullet_group.add(bullet)


running = True
while running:

    screen.fill(BG)
    # 先使用矩形碰撞检测，过滤掉大部分离enemy很远的子弹
    if pygame.sprite.spritecollide(bullet, enemy_group, False):
        color = BLUE
        # 如果进入enemy的矩形范围，使用Mask碰撞检测，判断是否有碰撞
        if pygame.sprite.spritecollide(bullet, enemy_group, True, pygame.sprite.collide_mask):
            color = RED
    else:
        color = GREEN
    bullet_group.update(color)
    
    enemy_group.draw(screen)
    bullet_group.draw(screen)
    pygame.draw.rect(screen, WHITE, enemy1.rect, 2)
    pygame.draw.rect(screen, WHITE, enemy2.rect, 2)
    pygame.draw.rect(screen, WHITE, enemy3.rect, 2)
    pygame.draw.rect(screen, WHITE, enemy4.rect, 2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
pygame.quit()
