import pygame

pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Mouse Demo")

radius = 10

running = True
while running:
    screen.fill((45, 68, 120))

    # print(pygame.mouse.get_pressed())

    # if pygame.mouse.get_pressed()[0]:
    #     print("Left mouse button pressed")
    # if pygame.mouse.get_pressed()[2]:
    #     print("Right mouse button pressed")

    # pygame.mouse.get_pressed() 方法只能检测左右键和滚轮的点击事件，
    # 滚轮滑动事件需要通过pygame.event.get() 方法来获取。
    # 对于有更多按键的鼠标也可以通过pygame.event.get() 方法来获取。

    if pygame.mouse.get_pressed()[0]:
        x, y = pygame.mouse.get_pos()
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 10)
        pygame.draw.circle(screen, (0, 0, 0), (x, y), 5)
    
    if pygame.mouse.get_pressed()[2]:
        x, y = pygame.mouse.get_pos()
        pygame.draw.rect(screen, (0, 0, 0), (x, y, 40, 40))
        pygame.draw.rect(screen, (255, 255, 255), (x, y, 25, 25))

    if pygame.mouse.get_pressed()[1]:
        pos = pygame.mouse.get_pos()
        pygame.draw.circle(screen, (128, 128, 128), pos, radius)

    for event in pygame.event.get():
        # if event.type == pygame.MOUSEBUTTONDOWN:
        #     print("Clicked")
        #     print(event)
        # if event.type == pygame.MOUSEBUTTONUP:
        #     print("Released")
        #     print(event)

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 4:
                radius += 1
            elif event.button == 5:
                radius -= 1
                if radius < 10:
                    radius = 10

        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()

pygame.quit()