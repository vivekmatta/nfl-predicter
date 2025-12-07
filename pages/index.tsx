import { useState, useEffect } from 'react';
import Head from 'next/head';

interface Team {
  abbr: string;
  name: string;
}

const NFL_TEAMS: Team[] = [
  { abbr: 'ARI', name: 'Arizona Cardinals' },
  { abbr: 'ATL', name: 'Atlanta Falcons' },
  { abbr: 'BAL', name: 'Baltimore Ravens' },
  { abbr: 'BUF', name: 'Buffalo Bills' },
  { abbr: 'CAR', name: 'Carolina Panthers' },
  { abbr: 'CHI', name: 'Chicago Bears' },
  { abbr: 'CIN', name: 'Cincinnati Bengals' },
  { abbr: 'CLE', name: 'Cleveland Browns' },
  { abbr: 'DAL', name: 'Dallas Cowboys' },
  { abbr: 'DEN', name: 'Denver Broncos' },
  { abbr: 'DET', name: 'Detroit Lions' },
  { abbr: 'GB', name: 'Green Bay Packers' },
  { abbr: 'HOU', name: 'Houston Texans' },
  { abbr: 'IND', name: 'Indianapolis Colts' },
  { abbr: 'JAX', name: 'Jacksonville Jaguars' },
  { abbr: 'KC', name: 'Kansas City Chiefs' },
  { abbr: 'LAC', name: 'Los Angeles Chargers' },
  { abbr: 'LAR', name: 'Los Angeles Rams' },
  { abbr: 'LV', name: 'Las Vegas Raiders' },
  { abbr: 'MIA', name: 'Miami Dolphins' },
  { abbr: 'MIN', name: 'Minnesota Vikings' },
  { abbr: 'NE', name: 'New England Patriots' },
  { abbr: 'NO', name: 'New Orleans Saints' },
  { abbr: 'NYG', name: 'New York Giants' },
  { abbr: 'NYJ', name: 'New York Jets' },
  { abbr: 'PHI', name: 'Philadelphia Eagles' },
  { abbr: 'PIT', name: 'Pittsburgh Steelers' },
  { abbr: 'SEA', name: 'Seattle Seahawks' },
  { abbr: 'SF', name: 'San Francisco 49ers' },
  { abbr: 'TB', name: 'Tampa Bay Buccaneers' },
  { abbr: 'TEN', name: 'Tennessee Titans' },
  { abbr: 'WAS', name: 'Washington Commanders' },
];

interface WeatherData {
  temperature: number;
  windSpeed: number;
  humidity: number;
  condition: string;
  isRain: number;
  isSnow: number;
  isIndoor: number;
}

interface PredictionResult {
  winner: string;
  homeWinProb: number;
  model: string;
}

export default function Home() {
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [season, setSeason] = useState(new Date().getFullYear());
  const [week, setWeek] = useState(1);
  const [isPlayoff, setIsPlayoff] = useState(false);
  const [month, setMonth] = useState(11);
  const [dayOfWeek, setDayOfWeek] = useState(6); // 0=Monday, 6=Sunday
  const [spread, setSpread] = useState(-3.5);
  const [overUnder, setOverUnder] = useState(45.0);
  const [favoriteTeam, setFavoriteTeam] = useState('');
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [loadingWeather, setLoadingWeather] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshMessage, setRefreshMessage] = useState('');

  const fetchWeather = async () => {
    if (!homeTeam) {
      alert('Please select a home team first');
      return;
    }
    setLoadingWeather(true);
    try {
      // Find the team object to get the abbreviation
      const teamObj = NFL_TEAMS.find(t => t.name === homeTeam);
      if (!teamObj) {
        alert('Invalid team selected');
        setLoadingWeather(false);
        return;
      }
      const response = await fetch(`http://localhost:5000/api/weather?team=${teamObj.abbr}`);
      const data = await response.json();
      if (data.error) {
        alert(`Error: ${data.error}`);
      } else {
        setWeather(data);
      }
    } catch (error) {
      alert(`Error fetching weather: ${error}`);
    } finally {
      setLoadingWeather(false);
    }
  };

  const handlePredict = async () => {
    if (!homeTeam || !awayTeam || !weather) {
      alert('Please fill in all fields and fetch weather first');
      return;
    }
    setLoadingPrediction(true);
    try {
      // Get team names (they're already stored as full names)
      const homeTeamName = homeTeam;
      const awayTeamName = awayTeam;
      // Get favorite team abbreviation
      const favTeamObj = favoriteTeam 
        ? NFL_TEAMS.find(t => t.abbr === favoriteTeam)
        : NFL_TEAMS.find(t => t.name === homeTeam);
      const favTeamAbbr = favTeamObj ? favTeamObj.abbr : '';
      
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          homeTeam: homeTeamName,
          awayTeam: awayTeamName,
          season,
          week,
          isPlayoff,
          month,
          dayOfWeek,
          spread,
          overUnder,
          favoriteTeam: favTeamAbbr,
          temperature: weather.temperature,
          windSpeed: weather.windSpeed,
          humidity: weather.humidity,
          isIndoor: weather.isIndoor,
          isRain: weather.isRain,
          isSnow: weather.isSnow,
        }),
      });
      const data = await response.json();
      if (data.error) {
        alert(`Error: ${data.error}`);
      } else {
        setPrediction(data);
      }
    } catch (error) {
      alert(`Error making prediction: ${error}`);
    } finally {
      setLoadingPrediction(false);
    }
  };

  const handleRefreshModel = async () => {
    setRefreshing(true);
    setRefreshMessage('Downloading latest data from Kaggle and training model...');
    try {
      const response = await fetch('http://localhost:5000/api/refresh_model', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.success) {
        const sourceInfo = data.source ? ` (from ${data.source})` : '';
        setRefreshMessage(`✅ ${data.message}${sourceInfo} - Model: ${data.model_name}, Samples: ${data.training_samples}`);
      } else {
        // Format error message with line breaks
        const errorMsg = data.error || 'Unknown error';
        setRefreshMessage(`❌ ${errorMsg.replace(/\n/g, ' ')}`);
      }
    } catch (error: any) {
      setRefreshMessage(`❌ Error: ${error.message || error}`);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    if (homeTeam && homeTeam.trim() !== '') {
      // Only fetch weather if we have a valid team selected
      const teamObj = NFL_TEAMS.find(t => t.name === homeTeam);
      if (teamObj) {
        fetchWeather();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [homeTeam]);

  return (
    <>
      <Head>
        <title>NFL Game Predictor</title>
        <meta name="description" content="Predict NFL game outcomes using machine learning" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '2rem' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ background: 'white', borderRadius: '12px', padding: '2rem', boxShadow: '0 10px 40px rgba(0,0,0,0.1)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
              <h1 style={{ margin: 0, fontSize: '2.5rem', color: '#333' }}>🏈 NFL Game Predictor</h1>
              <button
                onClick={handleRefreshModel}
                disabled={refreshing}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: refreshing ? '#ccc' : '#667eea',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: refreshing ? 'not-allowed' : 'pointer',
                  fontSize: '1rem',
                  fontWeight: 'bold',
                }}
              >
                {refreshing ? 'Refreshing...' : '🔄 Refresh Model'}
              </button>
            </div>
            {refreshMessage && (
              <div style={{
                padding: '1rem',
                marginBottom: '1rem',
                borderRadius: '8px',
                background: refreshMessage.includes('✅') ? '#d4edda' : '#f8d7da',
                color: refreshMessage.includes('✅') ? '#155724' : '#721c24',
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word',
                maxHeight: '200px',
                overflowY: 'auto',
              }}>
                {refreshMessage}
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Home Team</label>
                <select
                  value={homeTeam}
                  onChange={(e) => setHomeTeam(e.target.value)}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                >
                  <option value="">Select Home Team</option>
                  {NFL_TEAMS.map(team => (
                    <option key={team.abbr} value={team.name}>{team.abbr} - {team.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Away Team</label>
                <select
                  value={awayTeam}
                  onChange={(e) => setAwayTeam(e.target.value)}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                >
                  <option value="">Select Away Team</option>
                  {NFL_TEAMS.map(team => (
                    <option key={team.abbr} value={team.name}>{team.abbr} - {team.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Season</label>
                <input
                  type="number"
                  value={season}
                  onChange={(e) => setSeason(parseInt(e.target.value))}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Week</label>
                <input
                  type="number"
                  value={week}
                  onChange={(e) => setWeek(parseInt(e.target.value))}
                  min="1"
                  max="21"
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Month (1-12)</label>
                <input
                  type="number"
                  value={month}
                  onChange={(e) => setMonth(parseInt(e.target.value))}
                  min="1"
                  max="12"
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Day of Week (0=Mon, 6=Sun)</label>
                <input
                  type="number"
                  value={dayOfWeek}
                  onChange={(e) => setDayOfWeek(parseInt(e.target.value))}
                  min="0"
                  max="6"
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Spread (for favorite)</label>
                <input
                  type="number"
                  step="0.5"
                  value={spread}
                  onChange={(e) => setSpread(parseFloat(e.target.value))}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Over/Under</label>
                <input
                  type="number"
                  step="0.5"
                  value={overUnder}
                  onChange={(e) => setOverUnder(parseFloat(e.target.value))}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Favorite Team (optional)</label>
                <select
                  value={favoriteTeam}
                  onChange={(e) => setFavoriteTeam(e.target.value)}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid #ddd' }}
                >
                  <option value="">Auto (Home Team)</option>
                  {NFL_TEAMS.map(team => (
                    <option key={team.abbr} value={team.abbr}>{team.abbr} - {team.name}</option>
                  ))}
                </select>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <input
                  type="checkbox"
                  id="playoff"
                  checked={isPlayoff}
                  onChange={(e) => setIsPlayoff(e.target.checked)}
                  style={{ width: '20px', height: '20px' }}
                />
                <label htmlFor="playoff" style={{ fontWeight: 'bold' }}>Playoff Game</label>
              </div>
            </div>

            {weather && (
              <div style={{
                marginTop: '2rem',
                padding: '1.5rem',
                background: '#f8f9fa',
                borderRadius: '8px',
                border: '1px solid #dee2e6',
              }}>
                <h3 style={{ marginTop: 0 }}>🌤️ Weather Conditions</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                  <div><strong>Temperature:</strong> {weather.temperature}°F</div>
                  <div><strong>Wind Speed:</strong> {weather.windSpeed} mph</div>
                  <div><strong>Humidity:</strong> {weather.humidity}%</div>
                  <div><strong>Condition:</strong> {weather.condition}</div>
                  <div><strong>Indoor:</strong> {weather.isIndoor ? 'Yes' : 'No'}</div>
                  <div><strong>Rain:</strong> {weather.isRain ? 'Yes' : 'No'}</div>
                  <div><strong>Snow:</strong> {weather.isSnow ? 'Yes' : 'No'}</div>
                </div>
                <button
                  onClick={fetchWeather}
                  disabled={loadingWeather || !homeTeam}
                  style={{
                    marginTop: '1rem',
                    padding: '0.5rem 1rem',
                    background: loadingWeather ? '#ccc' : '#28a745',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: loadingWeather ? 'not-allowed' : 'pointer',
                  }}
                >
                  {loadingWeather ? 'Loading...' : '🔄 Refresh Weather'}
                </button>
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={loadingPrediction || !homeTeam || !awayTeam || !weather}
              style={{
                marginTop: '2rem',
                width: '100%',
                padding: '1rem',
                background: loadingPrediction ? '#ccc' : '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1.2rem',
                fontWeight: 'bold',
                cursor: loadingPrediction ? 'not-allowed' : 'pointer',
              }}
            >
              {loadingPrediction ? 'Predicting...' : '🎯 Make Prediction'}
            </button>

            {prediction && (
              <div style={{
                marginTop: '2rem',
                padding: '2rem',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                borderRadius: '8px',
                color: 'white',
              }}>
                <h2 style={{ marginTop: 0, fontSize: '2rem' }}>Prediction Result</h2>
                <div style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>
                  <strong>Predicted Winner:</strong> {prediction.winner}
                </div>
                <div style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>
                  <strong>Home Win Probability:</strong> {(prediction.homeWinProb * 100).toFixed(2)}%
                </div>
                <div style={{ fontSize: '1rem', opacity: 0.9 }}>
                  <strong>Model Used:</strong> {prediction.model}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

