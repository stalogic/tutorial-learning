import pygame
import pickle
import random
from pathlib import Path
from pygame.locals import *
from pygame import mixer


pygame.mixer.pre_init(44100, -16, 2, 512)
mixer.init()
pygame.init()

clock = pygame.time.Clock()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
TILE_SIZE = 50
FPS = 60
MAX_LEVEL = 8
FONT = pygame.font.SysFont('Bauhaus 93', 70)
FONT_SCORE = pygame.font.SysFont('Bauhaus 93', 50)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Platformer')

game_over = 0
main_menu = True
level = 3
score = 0


sun_img = pygame.image.load('img/sun.png')
bg_img = pygame.image.load("img/sky.png")
restart_img = pygame.image.load('img/restart_btn.png')
start_img = pygame.image.load('img/start_btn.png')
exit_img = pygame.image.load('img/exit_btn.png')


pygame.mixer.music.load('img/music.wav')
pygame.mixer.music.play(-1, 0.0, 5000)
coin_fx = pygame.mixer.Sound('img/coin.wav')
coin_fx.set_volume(0.5)
jump_fx = pygame.mixer.Sound('img/jump.wav')
jump_fx.set_volume(0.5)
game_over_fx = pygame.mixer.Sound('img/game_over.wav')
game_over_fx.set_volume(0.5)

class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.clicked = False
        

    def draw(self):
        action = False
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                action = True
                self.clicked = True
        
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        screen.blit(self.image, self.rect)
        return action


def draw_text(text, font, text_col, x, y):
    text = font.render(text, True, text_col)
    text_rect = text.get_rect(center=(x, y))
    screen.blit(text, text_rect)

def draw_grid():
    for line in range(21):
        pygame.draw.line(screen, (255, 255, 255), (0, line * TILE_SIZE), (SCREEN_WIDTH, line * TILE_SIZE))
        pygame.draw.line(screen, (255, 255, 255), (line * TILE_SIZE, 0), (line * TILE_SIZE, SCREEN_HEIGHT))


class World(object):
    def __init__(self, data):
        self.tile_list = []

        dirt_img = pygame.image.load('img/dirt.png')
        grass_img = pygame.image.load('img/grass.png')

        for i, row in enumerate(data):
            for j, tile in enumerate(row):
                match tile:
                    case 1:
                        img = pygame.transform.scale(dirt_img, (TILE_SIZE, TILE_SIZE))
                        img_rect = img.get_rect()
                        img_rect.x = j * TILE_SIZE
                        img_rect.y = i * TILE_SIZE
                        self.tile_list.append((img, img_rect))
                    case 2:
                        img = pygame.transform.scale(grass_img, (TILE_SIZE, TILE_SIZE))
                        img_rect = img.get_rect()
                        img_rect.x = j * TILE_SIZE
                        img_rect.y = i * TILE_SIZE
                        self.tile_list.append((img, img_rect))
                    case 3:
                        blob = Enemy(j * TILE_SIZE, i * TILE_SIZE + 16)
                        blob_group.add(blob)
                    case 4:
                        platform = Platform(j * TILE_SIZE, i * TILE_SIZE, 1, 0)
                        platform_group.add(platform)
                    case 5:
                        platform = Platform(j * TILE_SIZE, i * TILE_SIZE, 0, 1)
                        platform_group.add(platform)
                    case 6:
                        lava = Lava(j * TILE_SIZE, i * TILE_SIZE + (TILE_SIZE // 2))
                        lava_group.add(lava)
                    case 7:
                        coin = Coin(j * TILE_SIZE + (TILE_SIZE // 2), i * TILE_SIZE - (TILE_SIZE // 2))
                        coin_group.add(coin)
                    case 8:
                        exit = Exit(j * TILE_SIZE, i * TILE_SIZE - (TILE_SIZE // 2))
                        exit_group.add(exit)
                    case _:
                        continue

                

    def draw(self):
        for tile in self.tile_list:
            screen.blit(tile[0], tile[1])
            # pygame.draw.rect(screen, (255, 255, 255), tile[1], 2)

class Player(object):
    def __init__(self, x, y):
        self.reset(x, y)

    def reset(self, x, y):
        self.images_right = []
        self.images_left = []
        self.index = 0
        self.counter = 0
        self.dead_image = pygame.image.load('img/ghost.png')
        for num in range(1, 5):
            img = pygame.image.load(f'img/guy{num}.png')
            img = pygame.transform.scale(img, (40, 80))
            img_left = pygame.transform.flip(img, True, False)
            self.images_right.append(img)
            self.images_left.append(img_left)
        self.image = self.images_right[self.index]
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel_y = 0
        self.jumped = False
        self.direction = 0
        self.in_air = True

    def update(self, game_over):
        if game_over == 0:
            dx, dy = 0, 0
            key = pygame.key.get_pressed()
            if (key[pygame.K_w] or key[pygame.K_SPACE]) and not self.jumped and not self.in_air:
                jump_fx.play()
                self.vel_y = -15
                self.jumped = True
            if (key[pygame.K_w] or key[pygame.K_SPACE]):
                self.jumped = False
            if key[pygame.K_a]:
                dx -= 7
                self.counter += 1
                self.direction = -1
            elif key[pygame.K_d]:
                dx += 7
                self.counter += 1
                self.direction = 1

            if not key[pygame.K_a] and not key[pygame.K_d]:
                self.counter = 0
                self.index = 0
                self.image = self.images_right[self.index]
                if self.direction == 1:
                    self.image = self.images_right[self.index]
                elif self.direction == -1:
                    self.image = self.images_left[self.index]
            

            if self.counter > 7:
                self.counter = 0

                self.index += 1
                if self.index >= len(self.images_right):
                    self.index = 0
                if self.direction == 1:
                    self.image = self.images_right[self.index]
                elif self.direction == -1:
                    self.image = self.images_left[self.index]

            dy += self.vel_y
            self.vel_y += 1
            if self.vel_y > 10:
                self.vel_y = 10
    
            self.in_air = True
            for tile in world.tile_list:
                if tile[1].colliderect(self.rect.x, self.rect.y + dy, self.rect.width, self.rect.height):
                    if self.vel_y < 0:
                        self.vel_y = 0
                        dy = tile[1].bottom - self.rect.top
                    else:
                        self.vel_y = 0
                        dy = tile[1].top - self.rect.bottom
                        self.in_air = False
                if tile[1].colliderect(self.rect.x + dx, self.rect.y, self.rect.width, self.rect.height):
                    if self.direction == 1:
                        dx = tile[1].left - self.rect.right
                    elif self.direction == -1:
                        dx = tile[1].right - self.rect.left

            if pygame.sprite.spritecollide(player, lava_group, False):
                game_over_fx.play()
                game_over = -1
            
            if pygame.sprite.spritecollide(player, blob_group, False):
                game_over_fx.play()
                game_over = -1

            if pygame.sprite.spritecollide(player, exit_group, False):
                game_over = 1

            self.rect.x += dx
            self.rect.y += dy

            if self.rect.bottom > SCREEN_HEIGHT:
                self.rect.bottom = SCREEN_HEIGHT

        if game_over == -1:
            self.image = self.dead_image
            if self.rect.y >= 100:
                self.rect.y -= 5
        screen.blit(self.image, self.rect)
        # pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        return game_over

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('img/blob.png')
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.move_direction = 1 if random.random() > 0.5 else -1
        self.move_counter = random.randint(-10, 10)

    def update(self):
        self.rect.x += self.move_direction
        self.move_counter += 1
        if abs(self.move_counter) > 50:
            self.move_direction *= -1
            self.move_counter *= -1

class Lava(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load('img/lava.png')
        self.image = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE // 2))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, move_x, move_y):
        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load('img/platform.png')
        self.image = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE // 2))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.move_counter = 0
        self.move_direction = 1 if random.random() > 0.5 else -1
        self.move_x = move_x
        self.move_y = move_y

    def update(self):
        self.rect.x += self.move_direction * self.move_x
        self.rect.y += self.move_direction * self.move_y
        self.move_counter += 1
        if abs(self.move_counter) > 50:
            self.move_direction *= -1
            self.move_counter *= -1

class Exit(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load('img/exit.png')
        self.image = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE * 1.5))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    
class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        img = pygame.image.load('img/coin.png')
        self.image = pygame.transform.scale(img, (TILE_SIZE // 2, TILE_SIZE // 2))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
    


player = Player(100, SCREEN_HEIGHT - 130)
blob_group = pygame.sprite.Group()
platform_group = pygame.sprite.Group()
lava_group = pygame.sprite.Group()
exit_group = pygame.sprite.Group()
coin_group = pygame.sprite.Group()


def reset_level(level):
    player.reset(100, SCREEN_HEIGHT - 130)
    blob_group.empty()
    platform_group.empty()
    lava_group.empty()
    exit_group.empty()
    coin_group.empty()
    score_coin = Coin(TILE_SIZE // 2, TILE_SIZE // 2)
    coin_group.add(score_coin)

    file_path = Path(f'world_data/level{level}_data')
    if file_path.exists():
        pickle_in = open(file_path, 'rb')
        world_data = pickle.load(pickle_in)
    world = World(world_data)
    return world

world = reset_level(level)
restart_button = Button(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 100, restart_img)
start_button = Button(SCREEN_WIDTH // 2 - 350, SCREEN_HEIGHT // 2, start_img)
exit_button = Button(SCREEN_WIDTH // 2 + 150, SCREEN_HEIGHT // 2, exit_img)

running = True

while running:
    clock.tick(FPS)

    screen.blit(bg_img, (0, 0))
    screen.blit(sun_img, (100, 100))

    if main_menu:
        if start_button.draw():
            main_menu = False
        if exit_button.draw():
            running = False
    
    else:   
        world.draw()

        if game_over == 0:
            blob_group.update()
            platform_group.update()

            if pygame.sprite.spritecollide(player, coin_group, True):
                coin_fx.play()
                score += 1
            draw_text(f"X {score}", FONT_SCORE, WHITE, TILE_SIZE+20, 25)

        blob_group.draw(screen)
        platform_group.draw(screen)
        lava_group.draw(screen)
        exit_group.draw(screen)
        coin_group.draw(screen)
        game_over = player.update(game_over)

        if game_over == -1:
            draw_text("GAME OVER", FONT, RED, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            if restart_button.draw():
                world = reset_level(level)
                game_over = 0
                score = 0
        elif game_over == 1:
            level += 1
            if level < MAX_LEVEL:
                world = reset_level(level)
                game_over = 0
            else:
                draw_text("YOU WIN!", FONT, BLUE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                if restart_button.draw():
                    level = 1
                    world = reset_level(level)
                    game_over = 0

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    pygame.display.update()

pygame.quit()