# This script translates the Inertia Matrix matrix, as derived in matlab, to python for later 
# use in the controller

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ur5e_M(q: ArrayLike) -> NDArray[np.float64]:
    #Return the 6×6 joint-space mass matrix M(q).
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size != 6:
        raise ValueError(f"q must contain exactly 6 values; received {q_arr.size}.")

    theta1 = q_arr[0]
    theta2 = q_arr[1]
    theta3 = q_arr[2]
    theta4 = q_arr[3]
    theta5 = q_arr[4]
    theta6 = q_arr[5]

    t2 = np.cos(theta2)
    t3 = np.cos(theta3)
    t4 = np.cos(theta4)
    t5 = np.cos(theta5)
    t6 = np.sin(theta2)
    t7 = np.sin(theta3)
    t8 = np.sin(theta4)
    t9 = np.sin(theta5)
    t10 = theta2*1.0
    t11 = theta2*2.0
    t12 = theta3*1.0
    t13 = theta3*2.0
    t14 = theta4*1.0
    t15 = theta5*-1.0
    t16 = theta5*1.0
    t17 = theta5*-2.0
    t18 = theta5*2.0
    t19 = theta2+theta3
    t20 = theta2+theta5
    t21 = theta3+theta4
    t39 = -theta5
    t43 = theta3/2.0
    t44 = theta4/2.0
    t45 = theta5/2.0
    t22 = t5**2
    t23 = t9**2
    t24 = t8*-7.296735886e-2
    t26 = t8*-1.4593471772e-1
    t28 = t5*-5.6619090105e-3
    t30 = t3*7.98841985e-1
    t31 = t5*2.0e-4
    t32 = t8*2.2423236873e-2
    t33 = t6*3.970545245e-1
    t34 = np.sin(t19)
    t35 = np.sin(t20)
    t36 = np.sin(t21)
    t37 = t19+theta4
    t38 = t19+theta5
    t49 = np.sin(t44)
    t50 = t15+t19
    t52 = t3*t8*-7.90696775e-2
    t54 = t4*t7*-7.90696775e-2
    t57 = t4*t9*2.2423236873e-2
    t58 = t4*t9*4.4846473746e-2
    t61 = t2*t9*-2.4298510125e-2
    t64 = t39+theta2
    t66 = t19+t39
    t81 = t3*t4*t9*2.4298510125e-2
    t85 = t7*t8*t9*-2.4298510125e-2
    t40 = np.cos(t37)
    t41 = np.sin(t38)
    t42 = t37+theta5
    t46 = t22*-3.684174405565e-3
    t48 = t23*3.684174405565e-3
    t51 = t32-5.6619090105e-3
    t56 = t34*1.759699057e-1
    t59 = t35*1.21492550625e-2
    t60 = t36*2.4298510125e-2
    t65 = np.sin(t64)
    t67 = np.sin(t50)
    t68 = t15+t37
    t69 = t17+t37
    t70 = t18+t37
    t87 = np.sin(t66)
    t88 = t37+t39
    t89 = t49**2
    t63 = np.cos(t42)
    t71 = t40*2.475967735e-2
    t72 = t41*-1.12116184365e-2
    t73 = t41*1.12116184365e-2
    t74 = t40*-3.884174405565e-3
    t76 = np.cos(t68)
    t77 = np.cos(t69)
    t78 = np.cos(t70)
    t90 = t67*1.12116184365e-2
    t91 = t5*t51
    t92 = t65*1.21492550625e-2
    t93 = np.cos(t88)
    t94 = t89*2.0
    t98 = t87*1.12116184365e-2
    t104 = t51+t60
    t111 = t24+t46+t52+t54+t57+t81+t85+2.0337824227565e-2
    t114 = t26+t30+t46+t52+t54+t58+t81+t85+6.54132824267565e-1
    t79 = t63*-9.79623612e-4
    t82 = t63*1.0e-4
    t83 = t63*-6.6415326225e-3
    t95 = t77*9.2104360139125e-4
    t96 = t78*-9.2104360139125e-4
    t99 = t76*-9.79623612e-4
    t101 = t93*-1.0e-4
    t103 = t93*6.6415326225e-3
    t105 = t94-1.0
    t106 = t5*t104
    t107 = t9*t105*-2.2423236873e-2
    t109 = t82+t101
    t112 = t71+t79+t95+t96+t103
    t113 = t61+t72+t74+t83+t90+t99
    t110 = t24+t48+t107+1.6653649822e-2
    t115 = t56+t73+t98+t112
    t116 = t33+t59+t92+t115
    mt1 = [t24-t36*7.90696775e-2-np.sin(t11+t15+t21)*1.21492550625e-2-np.sin(t11+t13+theta4)*7.296735886e-2+np.sin(t11+t21+theta5)*1.21492550625e-2-np.sin(t11+t13+t15+theta4)*1.12116184365e-2+np.sin(t11+t13+theta4+theta5)*1.12116184365e-2-np.sin(t11+t21)*7.90696775e-2-np.sin(t15+t21)*1.21492550625e-2-np.sin(t15+theta4)*1.12116184365e-2+np.sin(t21+theta5)*1.21492550625e-2+np.sin(theta4+theta5)*1.12116184365e-2-np.sin(t10+t12+t14+t45)**2*5.6619090105e-3-np.sin(t10+t43)**2*1.59768397-np.sin(t10)**2*1.4261246875-np.sin(t16)**2*1.8420872027825e-3+np.sin(t37)**2*1.46115626192175e-2+np.sin(t42)**2*9.2104360139125e-4-np.sin(t43)**2*1.59768397-np.sin(t45)**2*3.0484624938e-2+np.sin(t68)**2*9.2104360139125e-4+np.sin(t10+t12+t14-theta5*5.0e-1)**2*5.6619090105e-3-t34**2*6.3379500004e-1+3.835567586810265,t116,t115,t112,t113,t109,t116,t3*1.59768397+t26+t46+t58-t3*t8*1.58139355e-1-t4*t7*1.58139355e-1+t3*t4*t9*4.859702025e-2-t7*t8*t9*4.859702025e-2+2.080257511767565,t114,t111,t106,t31,t115,t114,t26+t48-t9*t105*4.4846473746e-2+6.50448649862e-1,t110,t91,t31,t112]
    mt2 = [t111,t110,t46+2.0337824227565e-2,t28,t31,t113,t106,t91,t28,3.884174405565e-3,0.0,t109,t31,t31,t31,0.0,2.0e-4]

    values = np.concatenate((np.asarray(mt1, dtype=float), np.asarray(mt2, dtype=float)))
    return np.reshape(values, (6, 6), order="F")
