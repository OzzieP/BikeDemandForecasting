using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using MySql.Data.MySqlClient;


using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using System.Web;
using System;
using System.Globalization;
//using Workshop.Models;

namespace PassantsForecasting
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext _mlContext = new MLContext();

            string ConnectionString = "Data Source=146.59.229.11;Initial Catalog=Workshop;User ID=admin;Password=EPSIworkshop2020*";
            string rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
            string modelPath = Path.Combine(rootDir, "Data", "MLModel.zip");

            string query = "SELECT f.matricule AS Feu, CAST(e.nbPassant as REAL) AS NbPassants, CAST(e.jour as REAL) AS Jour, CAST(e.numWeek as REAL) as Semaine " +
                "FROM etat e " +
                "INNER JOIN feu f ON e.idFeu = f.idFeu " +
                "WHERE e.idFeu = 17 AND e.numWeek = 47";

            DatabaseLoader loader = _mlContext.Data.CreateDatabaseLoader<ModelInput>();
            DatabaseSource databaseSource = new DatabaseSource(SqlClientFactory.Instance, ConnectionString, query);

            IDataView dataView = loader.Load(databaseSource);
            IDataView firstWeekData = _mlContext.Data.FilterRowsByColumn(dataView, "Jour", upperBound: 1);
            IDataView nextWeekData = _mlContext.Data.FilterRowsByColumn(dataView, "Jour", lowerBound: 1);

            var forecastingPipeline = _mlContext.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedPassants",
                inputColumnName: "NbPassants",
                windowSize: 7,
                seriesLength: 24,
                trainSize: 10079,
                horizon: 7,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: "LowerBoundPassants",
                confidenceUpperBoundColumn: "UpperBoundPassants");

            SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstWeekData);

            Evaluate(nextWeekData, forecaster, _mlContext);

            var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(_mlContext);
            forecastEngine.CheckPoint(_mlContext, modelPath);

            Forecast(nextWeekData, 4, forecastEngine, _mlContext);
        }

        static void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
        {
            IDataView predictions = model.Transform(testData);

            IEnumerable<float> actual = mlContext.Data.CreateEnumerable<ModelInput>(testData, true).Select(observed => observed.NbPassants);
            IEnumerable<float> forecast = mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true).Select(prediction => prediction.ForecastedPassants[0]);
            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");
            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
        }

        static void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
        {
            ModelOutput forecast = forecaster.Predict();

            IEnumerable<string> forecastOutput = mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                .Take(horizon)
                .Select((ModelInput passants, int index) =>
                {
                    //string jour = Enum.GetName(typeof(DayOfWeek), passants.Jour);
                    float semaine = passants.Semaine;
                    float jour = passants.Jour;
                    //DateTime horaire = passants.Horaire;
                    float actualPassants = passants.NbPassants;
                    float lowerEstimate = Math.Max(0, forecast.LowerBoundPassants[index]);
                    float estimate = forecast.ForecastedPassants[index];
                    float upperEstimate = forecast.UpperBoundPassants[index];
                    return $"Date: {jour}\n" +
                    $"Actual Passants: {actualPassants}\n" +
                    $"Lower Estimate: {lowerEstimate}\n" +
                    $"Forecast: {estimate}\n" +
                    $"Upper Estimate: {upperEstimate}\n";
                });

            Console.WriteLine("Passants Forecast");
            Console.WriteLine("---------------------");
            foreach (var prediction in forecastOutput)
            {
                Console.WriteLine(prediction);
            }
        }
    }
}
