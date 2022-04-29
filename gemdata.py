import numpy as np

"""Triangle of individual payments"""
IPtriangle = np.array([[28446,31963,37775,40418,44116,50294,49620,46410,48295,52590,58599,60361],
                   [ 29251,36106,40125, 44499 ,45490 , 48040 , 49991 , 49694 , 49354 , 50606 , 53743 , np.nan],
                   [ 12464, 13441, 12951, 15370, 15339, 17843, 19570, 20881, 18304, 18604, np.nan,np.nan],
                   [ 5144, 5868, 6034, 5594, 5478, 7035,  10047, 8202, 8833, np.nan,np.nan,np.nan],
                   [ 2727, 2882, 3010, 2616, 2541, 3934, 5750, 4714, np.nan,np.nan,np.nan,np.nan],
                   [ 2359, 2422,  1264, 1984, 2906, 2726, 3313,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [ 1334, 918, 1250, 2137, 1294, 2267,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [ 1238, 1076, 1135,  1184,  1124, np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [ 941, 734,  904, 873,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [860,458,559,np.nan,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [282,456,np.nan,np.nan,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [727,np.nan,np.nan,np.nan ,np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan]]).T



"""Triangle of claim payments numbers"""
in_triangle =np.array([[34433, 35475,  37004,  37038,  36849,  39171,  37492,  34188,  31308,  30357,  30717,  30590 ],
                   [13796, 13718,  13820,  13631,  13416,  12601,  12282, 12245, 10743,  10117,  11081 , np.nan],
                   [1589,  1501 ,  1527 ,  1463 ,  1564 ,  1592 ,  2057 ,  1938 ,  1908 ,  1611 , np.nan,np.nan],
                   [568, 548,  436,  500,  422,  559,  739,  761,  639 , np.nan,np.nan,np.nan],
                   [278,209,  194,  164,  182,  273,  391,  295 , np.nan,np.nan,np.nan,np.nan],
                   [152,  134,  72,  80,  107,  175,  287 ,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [104,  51,  46,  63,  81,  155 ,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [55,  44,  29,  40,  42 , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [30,  27,  17,  26 ,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [31,  20,  17 ,np.nan,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [18,  16 ,np.nan,np.nan,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [19 ,np.nan,np.nan,np.nan ,np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan]]).T

# CNtriangle=CNtriangle.T

"""Triangle of incurred numbers"""
cased_number_triangle = np.array([[19508, 18794,  18099,17565,  17207, 16775, 17413,  17714,  15417,  14126,15481,  15178 ],
                   [4038,  3310,  2854,  2732,  2642, 3170, 4396,  4325,  3872,  3751,  3931  , np.nan],
                   [1374, 1114,  838,  875,  869,  1310,  1900,  1885,  1807,  1895  , np.nan,np.nan],
                   [727, 529,  395,  395,  474,  758,  1056,  1006,  1050 , np.nan,np.nan,np.nan],
                   [445, 318, 196, 250, 318, 476, 615, 685  , np.nan,np.nan,np.nan,np.nan],
                   [289,  185,  139,  178,  216,  296,  325  ,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [180,  148, 97,  123,  141,  145 ,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [125,  105,  73,  89,  102 , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [98, 80, 69, 66  ,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [71, 58, 51  ,np.nan,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [54, 43 ,np.nan,np.nan,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [36 ,np.nan,np.nan,np.nan ,np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan]]).T

"""Triangle of cased amounts"""
cased_amount_triangle= np.array([[56551, 63344, 66954,  72768, 76338,  83493,  92136,  97059,  94658,  104836,  112265,  107822],
                   [26907, 25706,  25502,  26613,  29399,  37080,  44671, 47808,  44919,  48968,  47921 ,np.nan],
                   [12910,  12497,  13034,  13663,  16012,  19807,  25963,  26658,  26905,  29487  ,np.nan,np.nan],
                   [7836,  7797, 8177,  9695,10904, 14191, 17438, 17590,  18725 , np.nan,np.nan,np.nan],
                   [6201,  6088, 5767,  7363, 8774, 11144, 12034,  12339 , np.nan,np.nan,np.nan,np.nan],
                   [4555, 4289, 4623,6052,  6308, 8161, 7718 ,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [3806, 3743,3620, 4111, 5255, 4926 ,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [3189, 2915, 2547, 3060, 2845 , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [2792, 2252, 1804, 1901 ,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [1987, 1762, 1853 ,np.nan,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [1784, 1795 ,np.nan,np.nan,np.nan , np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan],
                   [1068 ,np.nan,np.nan,np.nan ,np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan]]).T



reported_=np.array([57133,57896,58721,58248,57785,59753,58772,54761,49832,47899,49511,48946])

claims_inflation= np.array([1.06,	1.053,	1.045,	1.038,	1.03,	1.03,	1.03,	1.03,	1.03,	1.03,	1.03, 1.03])

czj=np.array([.5,.75,1.,1.25,1.5,2.,2.5,3.,4.,5.,7.,.01])