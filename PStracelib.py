def Lenovo(a):
        '''this function extract data from a pess file in  desktop LENOVO a--> is the file name'''
        import pyautogui

        # open 
        pyautogui.PAUSE = 25
        pyautogui.click(x=257, y=1407, clicks=2, interval=0.1, button='left')
       
        pyautogui.PAUSE = 1.5
        pyautogui.click(x=245, y=52, clicks=1, interval=1, button='left')
        pyautogui.click(x=296, y=82, clicks=1, button='left')

        pyautogui.click(x=582, y=503, clicks=2, interval=1,button='left')

        pyautogui.PAUSE = 1.5
        pyautogui.click(x=748, y=401, clicks=2,interval=0.1, button='left')

        pyautogui.click(x=868, y=832, clicks=1, button='left')
        pyautogui.typewrite( a)

        pyautogui.click(x=1303, y=846, clicks=1, button='left')

        # Excel
        pyautogui.PAUSE = 50
        pyautogui.click(x=639, y=751, clicks=1, button='left')
        pyautogui.click(x=343, y=1408, clicks=1, button='left')
        pyautogui.PAUSE = 10
        pyautogui.click(x=693, y=187, clicks=1, button='left')
        pyautogui.click(x=874, y=1219, clicks=1, button='left')

        pyautogui.click(x=781, y=829, clicks=2, button='left')
        pyautogui.PAUSE = 3
        pyautogui.typewrite( a)
        pyautogui.PAUSE = 1.5
        pyautogui.click(x=1191, y=877, clicks=1, button='left')
        pyautogui.click(x=1884, y=182, clicks=1, button='left')

        # fitting mode
        pyautogui.PAUSE = 3
        pyautogui.click(x=1126, y=327, clicks=1, button='left')
        pyautogui.PAUSE = 1.5
        pyautogui.click(x=751, y=460, clicks=2, button='left')
        with pyautogui.hold('win'):
                        pyautogui.press('up')
        pyautogui.click(x=190, y=742, clicks=1, button='left')

        #circuit
        pyautogui.PAUSE = 1.5
        # pyautogui.click(x=190, y=742, clicks=1, button='left')
        pyautogui.moveTo(383, 738, duration=1)  # move mouse to XY coordinates over num_second seconds
        pyautogui.click(x=383, y=738, clicks=1, button='left')
        pyautogui.moveTo(1000, 20, duration=1)
        pyautogui.click(x=217, y=139, clicks=1, button='left')
        pyautogui.moveTo(1000, 20, duration=1)
        pyautogui.moveTo(419, 749, duration=1)  # move mouse to XY coordinates over num_second seconds
        pyautogui.click(x=419, y=749, clicks=1, button='left')
        pyautogui.click(x=677, y=120, clicks=1, button='left')
        pyautogui.moveTo(497, 683, duration=1)  # move mouse to XY coordinates over num_second seconds
        pyautogui.click(x=497, y=683, clicks=1, button='left')
        pyautogui.click(x=1545, y=115, clicks=1, button='left')
        pyautogui.moveTo(626,666, duration=1)
        pyautogui.click(x=626, y=666, clicks=1, button='left')
        pyautogui.click(x=1840, y=1333, clicks=1, button='left')
        pyautogui.moveTo(x=1230, y=687,duration=1)
        pyautogui.dragTo(297, 687,2, button='left')
        ###
        pyautogui.moveTo(1000, 20, duration=1)
        pyautogui.PAUSE = 8

        pyautogui.click(x=2089, y=1252, clicks=3, interval=2, button='left')

        #excel fit
        pyautogui.PAUSE = 50
        pyautogui.click(x=2108, y=681, clicks=1, button='left')
        pyautogui.click(x=343, y=1408, clicks=1, button='left')
        pyautogui.PAUSE = 10
        pyautogui.click(x=693, y=187, clicks=1, button='left')
        pyautogui.click(x=874, y=1219, clicks=1, button='left')

        pyautogui.click(x=781, y=829, clicks=2, button='left')
        pyautogui.PAUSE = 3
        pyautogui.typewrite( a + '_fit')
        pyautogui.PAUSE = 1.5
        pyautogui.click(x=1191, y=877, clicks=1, button='left')
        pyautogui.click(x=1884, y=182, clicks=1, button='left')
        pyautogui.PAUSE = 3

        pyautogui.click(x=394, y=1415,clicks=1,button='left')

        pyautogui.click(x=802, y=757,clicks=1,button='left')
        pyautogui.typewrite("Rs,Rct")
        pyautogui.press('enter')
        # Rs
        pyautogui.PAUSE = 1.5
        pyautogui.click(x=255, y=1415, clicks=1, button='left')
        pyautogui.click(x=487, y=1265, clicks=1, button='left')
        pyautogui.click(x=915, y=873, clicks=1, button='left')
        pyautogui.click(x=915, y=873, clicks=1, button='right')
        pyautogui.click(x=951, y=970, clicks=1, button='left')
        pyautogui.moveTo(411, 1403,duration=1)
        pyautogui.click(x=411, y=1403, clicks=1, button='left')

        with pyautogui.hold('ctrl'):
                        pyautogui.press('v')
        pyautogui.press(',')

        # Rct
        pyautogui.PAUSE = 1.5
        pyautogui.click(x=255, y=1415, clicks=1, button='left')
        pyautogui.click(x=487, y=1265, clicks=1, button='left')
        pyautogui.click(x=915, y=904, clicks=1, button='left')
        pyautogui.click(x=915, y=904, clicks=1, button='right')
        pyautogui.click(x=948, y=1001, clicks=1, button='left')
        pyautogui.moveTo(411, 1403,duration=1)
        pyautogui.click(x=411, y=1403, clicks=1, button='left')
        with pyautogui.hold('ctrl'):
                        pyautogui.press('v')
        #save notepad
        with pyautogui.hold('win'):
                pyautogui.press('up')
        pyautogui.click(x=2128, y=8,clicks=1, button='left')
        pyautogui.click(x=944, y=701,clicks=1, button='left')
        pyautogui.typewrite(a)
        pyautogui.press('enter')

        #close everything
        # fitting window
        pyautogui.click(x=255, y=1415, clicks=1, button='left')
        pyautogui.click(x=487, y=1265, clicks=1, button='left')
        pyautogui.click(x=2129, y=19, clicks=1, button='left')

        #PStrace
        pyautogui.click(x=257, y=1407, clicks=2, interval=0.1, button='left')
        pyautogui.click(x=2129, y=19, clicks=1, button='left')
        pyautogui.click(x=1039, y=798, clicks=1, button='left')

        print( f'{a} finished')
