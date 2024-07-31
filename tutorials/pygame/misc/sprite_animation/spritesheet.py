import pygame

class SpriteSheet:
    def __init__(self, img):
        self.sprite_sheet = img

    def get_image(self, frame, width, height, scale=1, color_key=None):
        image = pygame.Surface((width, height)).convert_alpha()
        image.blit(self.sprite_sheet, (0, 0), (frame * width, 0, width, height))
        image = pygame.transform.scale(image, (width * scale, height * scale))
        image.set_colorkey(color_key)
        return image