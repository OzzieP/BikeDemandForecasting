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
            var currentCulture = new CultureInfo("fr-FR");
            var weekNo = currentCulture.Calendar.GetWeekOfYear(
                            DateTime.Now,
                            currentCulture.DateTimeFormat.CalendarWeekRule,
                            currentCulture.DateTimeFormat.FirstDayOfWeek);

            MLContext _mlContext = new MLContext();

            string ConnectionString = "Data Source=146.59.229.11;Initial Catalog=Workshop;User ID=admin;Password=EPSIworkshop2020*";
            string rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
            string modelPath = Path.Combine(rootDir, "Data", "MLModel.zip");

            string query = "SELECT CAST(e.numWeek as REAL) AS Semaine, CAST(e.jour as REAL) AS Jour, f.matricule AS Feu, CAST(SUM(e.nbPassant) as REAL) as NbPassants FROM etat e INNER JOIN feu f ON e.idFeu = f.idFeu GROUP BY e.numWeek, e.jour, f.matricule";

            DatabaseLoader loader = _mlContext.Data.CreateDatabaseLoader<ModelInput>();
            DatabaseSource databaseSource = new DatabaseSource(SqlClientFactory.Instance, ConnectionString, query);

            IDataView dataView = loader.Load(databaseSource);
            IDataView firstWeekData = _mlContext.Data.FilterRowsByColumn(dataView, "Semaine", upperBound: weekNo);
            IDataView nextWeekData = _mlContext.Data.FilterRowsByColumn(dataView, "Semaine", lowerBound: weekNo);

            var forecastingPipeline = _mlContext.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedPassants",
                inputColumnName: "NbPassants",
                windowSize: 4,
                seriesLength: 7,
                trainSize: 40314,
                horizon: 4,
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
