from playwright.sync_api import sync_playwright
import re
import math 
import csv
import os 
import numpy as np

h = 6.62607015 * 10**-34 
c = 3 * 10** 8 
k = 1.38 * 10**-23
e = math.e 
B = 2.898 * 10**-3 

try:
    os.makedirs("lamps")
except:
    pass
def parse_lamp_data(html: str):
    power = re.search(r"Потребляемая мощность[:\s]*([\d]+)", html)
    color_temp = re.search(r"Тип диода[:\s]*([A-Za-z0-9\s\-]+)", html)
    luminous_flux = re.search(r"(Световой поток|Фотонный поток)[:\s]*([\d\s]+)", html)
    
    return {
        "power_w": int(power.group(1)) if power else None,
        "color_temp_k": color_temp.group(1).strip() if color_temp else None,
        "luminous_flux": luminous_flux.group(2).strip() if luminous_flux else None
    }

with sync_playwright() as p:
    browser = p.webkit.launch(headless=False)
    page = browser.new_page()

    try:
        page.route("**/*.{png,jpg,jpeg,css}", lambda route: route.abort())
        page.goto("https://minifermer.ru/category/c208/", timeout=0)
        
        # Ждём появления блока с характеристиками
        page.wait_for_selector("ul", timeout=10000)
        links = page.eval_on_selector_all("a", "elements => elements.map(el => el.href)")
        products = []
        for i in links:
            if "product" in i:
                products.append(i)
        pribors = []
        for i in products:
            page.goto(i, timeout=0)
            page.wait_for_selector("ul", timeout=10000)
            html = page.content()
            data = parse_lamp_data(html)
            
            pribors.append(data)
        for i in range(len(pribors)):
            d = [("lenght", "intensive", "cumulative")]
            with open (f"lamps/{i}.csv", 'a', newline='') as f:
                csv.writer(f).writerows(d)
            

            T = str(pribors[i]['color_temp_k'])
        
            
            T = 5000
            d = []
            for la in range(380, 1101):
                la *= 10**-9
                intensive = (2 * h * c**2 / la**5) * (1/(np.exp(h * c / (la * k * T)) - 1))
                photons = intensive * la * 5.03 * 10**15
                d.append((la, np.pi * intensive, photons))
                with open (f'lamps/{i}.csv', 'a', newline = "") as f:
                    csv.writer(f).writerows(d)
        
        

    except Exception as e:
        print(f"❌ Ошибка: {e}")

    finally:
        browser.close()