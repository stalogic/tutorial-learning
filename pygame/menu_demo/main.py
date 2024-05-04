import pygame
import button

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Main Menu")


# define font
font = pygame.font.SysFont("Microsoft JhengHei", 40)

# define colors
TEXT_COLOR = (255, 255, 255)

# load button images
resume_img = pygame.image.load("image/button_resume.png").convert_alpha()
option_img = pygame.image.load("image/button_option.png").convert_alpha()
quit_img = pygame.image.load("image/button_quit.png").convert_alpha()
video_img = pygame.image.load("image/button_video.png").convert_alpha()
audio_img = pygame.image.load("image/button_audio.png").convert_alpha()
keys_img = pygame.image.load("image/button_keys.png").convert_alpha()
back_img = pygame.image.load("image/button_back.png").convert_alpha()

# create button images
resume_button = button.Button(304, 125, resume_img, 1)
option_button = button.Button(304, 250, option_img, 1)
quit_button = button.Button(304, 375, quit_img, 1)
video_button = button.Button(304, 125, video_img, 1)
audio_button = button.Button(304, 250, audio_img, 1)
keys_button = button.Button(304, 375, keys_img, 1)
back_button = button.Button(304, 500, back_img, 1)

# define game variables
game_paused = False
menu_state = "main"

def draw_text(text, font, color, x, y):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))


running = True
while running:
    screen.fill((52, 78, 91))

    if game_paused:
        if menu_state == "main":
            if resume_button.draw(screen):
                game_paused = False
            if option_button.draw(screen):
                menu_state = "options"
            if quit_button.draw(screen):
                running = False
        else:
            if video_button.draw(screen):
                pass
            if audio_button.draw(screen):
                pass
            if keys_button.draw(screen):
                pass
            if back_button.draw(screen):
                menu_state = "main"
    else:
        draw_text("按空格键暂停", font, TEXT_COLOR, 380, 250)

    for event in pygame.event.get():
        match event.type:
            case pygame.KEYDOWN:
                match event.key:
                    case pygame.K_SPACE:
                        game_paused = True
            case pygame.QUIT:
                running = False
            
        
    pygame.display.update()

pygame.quit()


