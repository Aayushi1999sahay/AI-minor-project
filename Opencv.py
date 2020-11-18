{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Opencv.py",
      "provenance": [],
      "authorship_tag": "ABX9TyM94Z18/dDIuEn4tkEnDYH9",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Aayushi1999sahay/AI-minor-project/blob/main/Opencv.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qvQPR-cM4-9Z"
      },
      "source": [
        "from keras.models import load_model\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import cv2 as cv"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XxyotXtk6-_2"
      },
      "source": [
        "model = load_model('/content/model.h5')"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "q7x26Li56U_b"
      },
      "source": [
        "def predict_digit(img):\n",
        "    img = np.array(img)\n",
        "    #reshaping to support our model input and normalizing\n",
        "    img = img.reshape(1,28,28,1)\n",
        "    img = img/255.0\n",
        "    #predicting the class\n",
        "    res = model.predict([img])\n",
        "    return np.argmax(res), max(res)\n"
      ],
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2PliSYht69ok",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 252
        },
        "outputId": "8f831fdb-86c0-4e0e-959a-2bcce372170b"
      },
      "source": [
        "img = cv.imread('/content/three.png')\n",
        "gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) \n",
        "plt.xticks([]),plt.yticks([])\n",
        "plt.show()"
      ],
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWAAAADrCAYAAABXYUzjAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAADfklEQVR4nO3YsW1CUQxA0f+ijAB1/v6zwBDUyQ5OjyiCBLkSnFNaLlzdwmtmNgD+30d9AMC7EmCAiAADRAQYICLAABEBBoh83rN8OBxm3/cnnQLwms7n88/MHK/ndwV43/ftdDo97iqAN7DWutyae0EARAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiAgwQWTPz9+W1vrdtuzzvHICX9DUzx+vhXQEG4HG8IAAiAgwQEWCAiAADRAQYICLAABEBBogIMEBEgAEiv24jG3fvmQi6AAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zitlJhdjy9AK"
      },
      "source": [
        "digit,accuracy = predict_digit(gray_img)\n",
        "print('Digit = ',digit)\n",
        "print('Accuracy = ',accuracy*100,'%')"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}