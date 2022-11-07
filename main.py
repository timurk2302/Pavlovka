import pandas as pd
from sklearn.impute import KNNImputer
import openpyxl
import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt
import seaborn as sns
import time
from pysheds.grid import Grid

class Pavlovka():
    def __int__(self):
        #настройки отображения пандас
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None


        self.grid = Grid.from_raster('HGT1.tif') # Чтение цмр
        self.dem = self.grid.read_raster('HGT1.tif') # Чтение цмр
        flooded_dem,inflated_dem = self.Correction(self.dem,self.grid)
        f_dir = self.F_dir(inflated_dem)
        # f_inf = self.F_inf(inflated_dem)
        # df = self.Create_D_F(f_inf)
        # rec_df = self.Recovery(df)
        # self.Accumulation_Flow(f_dir)
        # Вывод во вне
        #rec_df.to_excel("test.xlsx", sheet_name="passengers", index=False)
        # print(rec_df)
        self.River_network(f_dir)


    def F_dir(self,inflated_dem):
        inflated_dem = inflated_dem
        return self.grid.flowdir(inflated_dem)

    def F_inf(self,inflated_dem):
        inflated_dem = inflated_dem
        return self.grid.flowdir(inflated_dem, routing='dinf') #алгоритм deterministic infi nity

    def Create_D_F(self,f_inf):
        f_inf = f_inf
        df=pd.DataFrame(f_inf)
        print(df.describe())
        return df

    def Recovery(self,df):
        df = df
        #Импутер восстановление данных
        imputer = KNNImputer(n_neighbors=8) #Метод
        rec_df = pd.DataFrame(imputer.fit_transform(df)) #Перезапись фрема с восстановлнеием
        return rec_df

    def Correction(self, dem, grid):
        dem = self.dem
        grid = self.grid

        self.flooded_dem = grid.fill_depressions(dem)  # Заполнение углублений

        self.inflated_dem = grid.resolve_flats(self.flooded_dem)  # Коррекция

        return (self.flooded_dem, self.inflated_dem)

    def Accumulation_Flow(self,f_dir):
        f_dir=f_dir
        acc = self.grid.accumulation(f_dir)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0)
        plt.grid('on', zorder=0)
        im = ax.imshow(acc, extent=self.grid.extent, zorder=2,
                       cmap='cubehelix',
                       norm=colors.LogNorm(1, acc.max()),
                       interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Высоты')
        plt.title('Накопление потока', size=14)
        plt.xlabel('Долгота')
        plt.ylabel('Широта')
        plt.tight_layout()
        plt.show()

    def River_network(self,f_dir):
        f_dir = f_dir
        # x, y = 56.624,55.544 #ТОчка сбора воды
        x, y = 56.535,55.413 # ТОчка сбора воды

        catch = self.grid.catchment(x=x, y=y, fdir=f_dir, xytype='coordinate')
        self.grid.clip_to(catch)
        acc = self.grid.accumulation(f_dir, apply_output_mask=False)

        acc_number = 0 #Количество областей водосборников, чем выше значение тем меньше сборников на выходе

        branches = self.grid.extract_river_network(f_dir, acc > acc_number,apply_output_mask=False) # Извлечение речной сети

        sns.set_palette('husl')
        fig, ax = plt.subplots(figsize=(8.5, 6.5))

        plt.xlim(self.grid.bbox[0], self.grid.bbox[2])
        plt.ylim(self.grid.bbox[1], self.grid.bbox[3])
        ax.set_aspect('equal')

        for branch in branches['features']:
            line = np.asarray(branch['geometry']['coordinates'])
            plt.plot(line[:, 0], line[:, 1])

        _ = plt.title(f'Сеть каналов потока >{acc_number}', size=14)
        ax.scatter(x,y,marker = 'X')
        ax.text(x,y,"Точка водосбора",fontsize = 14)
        plt.show()


A = Pavlovka()
A.__int__()



# x, y = -97.294167, 32.73750
#
# catch = grid.catchment(x=x, y=y, fdir=new_df, xytype='coordinate')
# grid.clip_to(catch)
# dist = grid.distance_to_outlet(x, y, fdir=new_df, xytype='coordinate')
#
# fig, ax = plt.subplots(figsize=(8,6))
# fig.patch.set_alpha(0)
# plt.grid('on', zorder=0)
# im = ax.imshow(dist, extent=grid.extent, zorder=2,
#                cmap='cubehelix_r')
# plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Distance to outlet', size=14)
# plt.show()