import numpy as np
import LinearRegression

class PolynomialRegression(LinearRegression.LinearRegression):
    """class untuk model polynomial 2"""
    x_train:"np.ndarray | list | tuple"
    y_train:"np.ndarray | list | tuple"
    weight:"np.ndarray | list | tuple"
    degree:int
    error:float
    alpha:float
    loss_function_type:str
    lost_function_list:list
    
    def __init__(self, 
            x_train=np.ndarray, 
            y_train=np.ndarray,  
            degree=2,
            alpha=0.1):
        """inisiasi awal class"""
        try:
            self.x_train = self.variableVerificator(x_train)
            self.y_train = self.variableVerificator(y_train)
            self.weight = self.variableVerificator(
                [1 for _ in range(degree+1)]).reshape(-1,1).astype(np.float16)
        except TypeError as e:
            print("tipe data untuk x_train atau y_train salah")
            print(e)

        """x_train = b0 + b1w1 + b2w2 + ... + bnwn, b0 = bias term"""
        self.x_train = self.xDegreeGenerator(x_train=x_train,degree=degree)
        self.alpha = alpha
        self.degree = degree

    def xDegreeGenerator(self,x_train=np.ndarray,degree=int) -> np.ndarray:
        """ubah x_train menjadi matrix varmound"""
        size = x_train.size
        ones = np.ones((x_train.size, 1))
        result = [x_train**i for i in range(1,degree+1)]
        result = np.array(result).ravel().reshape(degree,size).transpose()
        return np.concatenate([
            ones,
            result
        ], axis=1)
        
    def fit(
            self, 
            loss_function_type = "MAE",
            iter=500
        ) -> None:
        """
        fitting data ke model regresi linear

        Parameters :
        ----------

        cost_function:
        pilih cost function yang akan digunakan = MAE, RMSE, R-Squared, MSE
        """
        match loss_function_type:
            case "MAE" :
                self.loss_function = MAE(weight=self.weight,degree=self.degree,alpha=self.alpha)
            case "MSE":
                self.loss_function = MSE(weight=self.weight,degree=self.degree,alpha=self.alpha)
            case "RMSE":
                self.loss_function = RMSE(weight=self.weight,degree=self.degree,alpha=self.alpha)
            case "R-Squared":
                self.loss_function = RSquared(weight=self.weight,degree=self.degree,alpha=self.alpha)
            case _:
                print("tidak ada loss function dengan nama tersebut")

        for _ in range(iter):
            self.loss_function.update(
                x_train=self.x_train, y_train=self.y_train
            )

        self.weight = self.loss_function.returnParameter()

    def predict(self, x_test=np.ndarray) -> np.ndarray:
        return self.xDegreeGenerator(x_test,self.degree).dot(self.weight)

    def __str__(self):
        return (
            f"weight1 : \n {self.weight}"
        )

    def name(self):
        return self.loss_function.name()

    def __del__(self):
        self.x_train = None
        self.y_train = None
        self.weight = None
        self.degree = None
        self.error = None
        self.alpha = None
        self.loss_function_type = None
        self.lost_function_list = None

class LossFunction(LinearRegression.LossFunction):
    alpha:float
    weight:np.ndarray
    error:float

    def __init__(self, weight=np.ndarray, degree=int, alpha=0.01) -> None:
        """inisiasi class"""
        self.alpha = alpha
        self.weight = weight
        self.degree = degree

    def Prediction(self, x_train=np.ndarray) -> np.ndarray:
        """hasil prediksi dari model"""
        return x_train.dot(self.weight) 
    
    def actualToPredictionDistance(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """perbedaan antara nilai aktual dan nilai prediksi"""
        return y_train - self.Prediction(x_train=x_train)

    def absOfActualToPrediction(self,
        x_train=np.ndarray,
        y_train=np.ndarray) -> np.ndarray:
        """nilai absolute dari jarak antara nilai aktual dan nilai prediksi"""
        return np.abs(
            self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            )
        )
    
    def returnParameter(self) -> np.ndarray:
        """mengembalikan parameter ke model untuk proses prediksi"""
        return self.weight

    def predict(self, x_test=np.ndarray) -> np.ndarray:
        return x_test.dot(self.weight)

    def __del__(self):
        self.weight = None
        self.alpha = None

class MAE(LossFunction, LinearRegression.MAE):
    def __init__(self, weight=np.ndarray, degree=int, alpha=0.01):
        """inisiasi awal class"""
        super().__init__(weight, degree, alpha)

    def derivativeToWeight(self, x_train=np.ndarray, y_train=np.ndarray, degree=int):
        return np.mean(
            -1 * x_train[:,degree] * self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ) / self.absOfActualToPrediction(
                x_train=x_train, y_train=y_train
            )
        ) 
    
    def update(self, x_train=np.ndarray, y_train=np.ndarray) -> None:
        for i in range(self.degree+1):
            self.weight[i,0] = self.updateParameter(
                self.weight[i,0], self.derivativeToWeight(
                    x_train=x_train,y_train=y_train,degree=i)
            )

    def name(self):
        return "MAE Loss Function"

class MSE(LossFunction, LinearRegression.MSE):
    def __init__(self, weight=np.ndarray, degree=int, alpha=0.01):
        """inisiasi awal class"""
        super().__init__(weight, degree, alpha)

    def derivativeToWeight(self, x_train=np.ndarray, y_train=np.ndarray, degree=int):
        """turunan dari loss function MSE respectively ke weight"""
        return np.mean(
            -2 * x_train[:,degree] * self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            )
        )
    
    def update(self, x_train=np.ndarray, y_train=np.ndarray):
        for i in range(self.degree+1):
            self.weight[i,0] = self.updateParameter(
                self.weight[i,0], self.derivativeToWeight(
                    x_train=x_train,y_train=y_train,degree=i)
                )
                

    def name(self):
        return "MSE Loss Function"

class RMSE(LossFunction, LinearRegression.RMSE):
    def __init__(self, weight=np.ndarray, degree=int, alpha=0.01):
        """inisiasi awal class"""
        super().__init__(weight, degree, alpha)

    def loss(self, 
        x_train=np.ndarray, 
        y_train=np.ndarray) -> np.ndarray:
        """hitung Root Mean Square Error (RMSE) dari model"""
        return np.sqrt(
            np.mean(
                np.square(
                    self.actualToPredictionDistance(
                        x_train=x_train,y_train=y_train
                    )   
                )
            )
        )

    def derivativeToWeight(self, x_train=np.ndarray, y_train=np.ndarray, degree=int):
        """turunan dari loss function RMSE respectively ke weight"""
        return np.mean(
            -1 * x_train[:,degree] * np.divide(self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ),self.loss(
                x_train=x_train, y_train=y_train
            ))
        )
    
    def update(self, x_train=np.ndarray, y_train=np.ndarray):
        for i in range(self.degree+1):
            self.weight[i,0] = self.updateParameter(
                self.weight[i,0], self.derivativeToWeight(
                    x_train=x_train,y_train=y_train,degree=i)
            )

    def name(self):
        return "RMSE Loss Function"

class RSquared(LossFunction, LinearRegression.RSquared):
    def __init__(self, weight=np.ndarray, degree=int, alpha=0.01):
        super().__init__(weight, degree, alpha)

    def derivativeToWeight(self, x_train=np.ndarray, y_train=np.ndarray, degree=int):
        """turunan dari loss function R-Squared respectively ke weight"""
        y_mean = np.mean(y_train)
        return np.mean(
            -2 * x_train[:,degree] * np.divide(self.actualToPredictionDistance(
                x_train=x_train, y_train=y_train
            ),np.square(
                y_train - y_mean
            ))
        )
    
    def update(self, x_train=np.ndarray, y_train=np.ndarray):
        for i in range(self.degree+1):
            self.weight[i,0] = self.updateParameter(
                self.weight[i,0], self.derivativeToWeight(
                    x_train=x_train,y_train=y_train,degree=i)
            )

    def name(self):
        return "R-Squared Loss Function"