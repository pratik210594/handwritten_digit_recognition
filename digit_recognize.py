{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2311c143",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import fetch_openml"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "94ae1aa6",
   "metadata": {},
   "outputs": [],
   "source": [
    "mnist = fetch_openml('mnist_784')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "469d79c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "x ,y = mnist['data'], mnist['target']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f355b04c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(70000, 784)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "dd1da5ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(70000,)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "76140c5b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Matplotlib is building the font cache; this may take a moment.\n"
     ]
    }
   ],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "32c6ff0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "b739ced8",
   "metadata": {},
   "outputs": [],
   "source": [
    "some_digit = x.to_numpy()[36001]\n",
    "some_digit_image = some_digit.reshape(28, 28) #reshape to plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "2a28a238",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(-0.5, 27.5, 27.5, -0.5)"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation =\"nearest\")\n",
    "plt.axis(\"off\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "87279dbe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'2'"
      ]
     },
     "execution_count": 149,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y[36001]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "id": "eb01dc6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test = x.to_numpy()[0:6000], x.to_numpy()[6000:7000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "6380206a",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train, y_test = y.to_numpy()[0:6000], y.to_numpy()[6000:7000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "id": "159f717a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "shuffle_index = np.random.permutation(6000)\n",
    "x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7eb9faca",
   "metadata": {},
   "source": [
    "## create a 2 detector "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "id": "03361795",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train = y_train.astype(np.int8)\n",
    "y_test = y_test.astype(np.int8)\n",
    "y_train_2 = (y_train<2)\n",
    "y_test_2 = (y_test==2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "64f834b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([9, 6, 3, ..., 7, 9, 3], dtype=int8)"
      ]
     },
     "execution_count": 175,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "id": "e5b6275d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "id": "30674053",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = LogisticRegression(tol = 0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "id": "e4133086",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\pc\\appdata\\local\\programs\\python\\python38\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-17 {color: black;background-color: white;}#sk-container-id-17 pre{padding: 0;}#sk-container-id-17 div.sk-toggleable {background-color: white;}#sk-container-id-17 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-17 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-17 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-17 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-17 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-17 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-17 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-17 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-17 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-17 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-17 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-17 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-17 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-17 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-17 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-17 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-17 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-17 div.sk-item {position: relative;z-index: 1;}#sk-container-id-17 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-17 div.sk-item::before, #sk-container-id-17 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-17 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-17 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-17 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-17 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-17 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-17 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-17 div.sk-label-container {text-align: center;}#sk-container-id-17 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-17 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-17\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(tol=0.1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-17\" type=\"checkbox\" checked><label for=\"sk-estimator-id-17\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(tol=0.1)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(tol=0.1)"
      ]
     },
     "execution_count": 178,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.fit(x_train, y_train_2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "id": "97390b49",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False])"
      ]
     },
     "execution_count": 179,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.predict([some_digit])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "06f95a65",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8rc1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
