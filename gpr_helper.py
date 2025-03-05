import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Creating GPR class with gpytorch
class GPRModel(gpytorch.models.ExactGP):
    """
    Gaussian Process Regression (GPR) model.

    Parameters:
    ----------
    train_x : torch.Tensor
        Training input data.
    train_y : torch.Tensor
        Training target data.
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function.
    kernel_type : str
        Type of kernel to use ('rbf', 'matern', 'rq', etc.).
    **kernel_params : dict
        Additional parameters for the kernel.

    Attributes:
    ----------
    kernel : gpytorch.kernels.Kernel
        The kernel used for the GPR model.
    mean_module : gpytorch.means.Mean
        The mean function for the GPR model.
    covar_module : gpytorch.kernels.Kernel
        The covariance function for the GPR model.
    """
    def __init__(self, train_x, train_y, likelihood, kernel_type, **kernel_params):
        """
        Initializes the Gaussian Process Regression (GPR) model with given training data, likelihood, and kernel type.

        Parameters:
        ----------
        train_x : torch.Tensor
            Training input features.
        train_y : torch.Tensor
            Training target values.
        likelihood : gpytorch.likelihoods.Likelihood
            The likelihood function to be used in the GPR model.
        kernel_type : str
            The type of kernel to be used ('rbf', 'matern', 'rq', 'rbf_matern', 'rbf_rq', or 'matern_rq').
        **kernel_params : dict
            Additional parameters for the kernel, including 'noise' for noise level.

        Raises:
        -------
        ValueError
            If an unknown kernel type is provided.
        """
        super().__init__(train_x, train_y, likelihood)

        # Extract noise and kernel parameters
        noise = kernel_params.pop('noise', 0.01)
        likelihood.initialize(noise=noise)

        # Kernel is choosen based on kernel_type
        if kernel_type == 'rbf':
            self.kernel = gpytorch.kernels.RBFKernel(**kernel_params)
        elif kernel_type == 'matern':
            self.kernel = gpytorch.kernels.MaternKernel(nu=2.5, **kernel_params)
        elif kernel_type == 'rq':
            self.kernel = gpytorch.kernels.RQKernel(**kernel_params)
        elif kernel_type == 'rbf_matern':
            rbf_kernel = gpytorch.kernels.RBFKernel(**kernel_params)
            matern_kernel = gpytorch.kernels.MaternKernel(nu=0.5, **kernel_params)
            self.kernel = gpytorch.kernels.AdditiveKernel(rbf_kernel, matern_kernel)
        elif kernel_type == 'rbf_rq':
            rbf_kernel = gpytorch.kernels.RBFKernel(**kernel_params)
            rq_kernel = gpytorch.kernels.RQKernel(**kernel_params)
            self.kernel = gpytorch.kernels.AdditiveKernel(rbf_kernel, rq_kernel)
        elif kernel_type == 'matern_rq':
            matern_kernel = gpytorch.kernels.MaternKernel(nu=1.5, **kernel_params)
            rq_kernel = gpytorch.kernels.RQKernel(**kernel_params)
            self.kernel = gpytorch.kernels.AdditiveKernel(matern_kernel, rq_kernel)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Generated with the help of AI assisstent
        # Set up the mean and covariance module
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.kernel)

    # Generated with the help of AI assisstent
    def forward(self, x):
        """
        Forward pass through the GPR model.

        Parameters:
        ----------
        x : torch.Tensor
            Input data.

        Returns:
        -------
        gpytorch.distributions.MultivariateNormal
            The predicted distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# To plot Predicted vs Actual Plot
def actual_predicted_plot(test_y, y_pred, lower, upper):
    plt.figure(figsize=(8, 6))

    # Scatter plot for actual vs predicted values
    sns.scatterplot(x=test_y, y=y_pred, alpha=0.6, edgecolor=None)
    # Plot the ideal line (where predicted equals actual)
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], '--', color='red')

    # Generated with the help of AI assisstent
    plt.fill_between(test_y, lower, upper, color='gray', alpha=0.2, label="Confidence Interval")

    # Labels and title
    plt.xlabel("Actual Days to Death")
    plt.ylabel("Predicted Days to Death")
    plt.title("Actual vs. Predicted Days to Death (GPR)")

    plt.legend()
    plt.show()


# To plot distribution plot of prediction and actual
def distribution_plot(y_true, y_pred):
    """
    Plot actual vs. predicted values with confidence intervals.

    Parameters:
    ----------
    test_y : torch.Tensor
        Actual target values.
    y_pred : torch.Tensor
        Predicted values.
    lower : torch.Tensor
        Lower bounds of the confidence intervals.
    upper : torch.Tensor
        Upper bounds of the confidence intervals.
    """
    plt.figure(figsize=(10, 6))

    # Plot distribution of true vs predicted values
    sns.kdeplot(y_true, label='True Values', color='blue', fill=True)
    sns.kdeplot(y_pred, label='Predicted Values', color='red', fill=True)

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of True vs Predicted Values')

    plt.legend()
    plt.show()


# Function to train and evaluate the model
def train_and_evaluate_gpr_model(kernel_type, kernel_params, train_x, train_y, test_x, test_y, likelihood):
    """
    Train and evaluate the GPR model.

    Parameters:
    ----------
    kernel_type : str
        Type of kernel to use.
    kernel_params : dict
        Additional parameters for the kernel.
    train_x : torch.Tensor
        Training input data.
    train_y : torch.Tensor
        Training target data.
    test_x : torch.Tensor
        Test input data.
    test_y : torch.Tensor
        Test target data.
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function.

    Returns:
    -------
    tuple
        Predicted values, lower and upper bounds of confidence intervals, MSE, MAE, R2 score.
    """
    # Check if GPU is available and move data and model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    # Add numerical stabilization settings
    # "With" was Generated with the help of AI assisstent
    # to overcome run time error related gpu memory limit
    with gpytorch.settings.cholesky_jitter(1e-4), \
        gpytorch.settings.max_cg_iterations(1000), \
        gpytorch.settings.fast_computations(covar_root_decomposition=True), \
        gpytorch.settings.skip_posterior_variances(False):  # Ensure stable variance calculations

        # Move data to the device   
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        test_x = test_x.to(device)
        test_y = test_y.to(device)

        # Initialize the model and move it to the device
        model = GPRModel(train_x, train_y, likelihood, kernel_type=kernel_type, **kernel_params).to(device)

        # Add jitter to the kernel for numerical stability
        model.covar_module.register_prior(
            "jitter_prior",
            gpytorch.priors.NormalPrior(1e-6, 0.01),
            lambda module: module.jitter
        )
        model.covar_module.jitter = 1e-6  # Small diagonal addition

        # Set the model to training mode
        model.train()
        likelihood.train()

        # Setting up the optimizer, learning rate scheduler, and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        best_loss = float('inf')
        patience = 19 # Used 19 by trail and error, as it gave optimal results
        epochs_since_improvement = 0

        for i in range(100):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # Update learning rate

            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if i % 20 == 0:
                print(f"Iteration {i+1}/{100} - Loss: {loss.item()}")

            # Early stopping to prevent from overfitting
            if epochs_since_improvement >= patience:
                print("Early stopping")
                break

        # Switch model and likelihood to evaluation mode
        model.eval()
        likelihood.eval()

        # Make predictions on the model
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            y_pred = observed_pred.mean.cpu()
            lower, upper = observed_pred.confidence_region()
            lower = lower.cpu()
            upper = upper.cpu()

        # Calculate Mean Squared Error, Mean Absolute Error, and R2
        mse = mean_squared_error(test_y.cpu().numpy(), y_pred.cpu().numpy())
        mae = mean_absolute_error(test_y.cpu().numpy(), y_pred.cpu().numpy())
        r2 = r2_score(test_y.cpu().numpy(), y_pred.cpu().numpy())
        rmse = np.sqrt(mse)
        print(f'MSE: {mse}, MAE: {mae}, R2: {r2}, RMSE: {rmse}')

        return y_pred, lower, upper, mse, mae, r2


# Grid search for best kernel parameters
def grid_search_kernel_params(kernel_types, train_x, train_y, test_x, test_y, param_grid):
    """
    Perform grid search to find the best kernel parameters.

    Parameters:
    ----------
    kernel_types : list of str
        List of kernel types to try.
    train_x : torch.Tensor
        Training input data.
    train_y : torch.Tensor
        Training target data.
    test_x : torch.Tensor
        Test input data.
    test_y : torch.Tensor
        Test target data.
    param_grid : list of dict
        List of parameter grids to try.

    Returns:
    -------
    tuple
        Best predicted values, lower and upper bounds, kernel type, kernel parameters, MSE, MAE, R2 score
    """
    best_mse = float('inf')
    best_kernel_params = None
    best_kernel_type = None
    best_mae = 0
    best_r2 = 0
    best_y_pred = None
    best_lower = None
    best_upper = None

    torch.cuda.empty_cache()
    for kernel_type in kernel_types:
        for params in param_grid:
            torch.cuda.empty_cache()  
            
            # Using GaussianLikelihoodWithMissingObs to handle potential missing values in the dataset
            # If no missing value, it will still function similarly to GaussianLikelihood()
            likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs()
            noise = params.pop('noise', 0.01)
            
            # Initialize likelihood with noise
            likelihood.initialize(noise=noise)

            print(f"Trying kernel type: {kernel_type} with parameters: {params} noise - {noise}")

            y_pred, lower, upper, mse, mae, r2 = train_and_evaluate_gpr_model(
                kernel_type, params, train_x, train_y, test_x, test_y, likelihood
            )

            # If mse of latest trained kernel and kernel paramter is < best_mse
            # Updating the kernel and kernel paramter combination to the latest
            if mse < best_mse:
                best_mse = mse
                best_kernel_params = {**params, 'noise': noise}
                best_kernel_type = kernel_type
                best_mae = mae
                best_r2 = r2
                best_y_pred = y_pred
                best_lower = lower
                best_upper = upper

    print(f"Best Kernel Type: {best_kernel_type}, Best Parameters: {best_kernel_params}, MSE: {best_mse}")

    return best_y_pred, best_lower, best_upper, best_kernel_type, best_kernel_params, best_mse, best_mae, best_r2


# Docstrings generated by: Microsoft Copilot: Your AI companion. (2025).
# Microsoft Copilot: Your AI companion. [online] Available at:
# https://copilot.microsoft.com/chats/XpnpvuQBKkwBheHuyRpeV
# [Accessed 23 Feb. 2025].
