# Perudo Simple Web Interface - MVP Specification

This document outlines a minimal web interface for the Perudo game following MVP principles for up to 20 users, prioritizing simplicity and Test-Driven Development (TDD).

## Design Principles

- **MVP First**: Core gameplay only, no advanced features
- **Maximum 20 Users**: Simple architecture sufficient for small user base  
- **No Advanced Security**: Basic validation only, no enterprise security features
- **Keep It Simple**: Minimal dependencies, straightforward implementation
- **TDD Compliance**: Follow existing project testing patterns

## Target User Scenarios

1. **Simple Room Creation**: Enter nickname, create room with 4-digit code
2. **Easy Room Joining**: Join via room code  
3. **Basic Gameplay**: Turn-based play with existing AI agents
4. **Game Results**: Simple win/loss display, restart option

## Technical Architecture

### Simple Tech Stack

**Backend**: Flask (Python) - minimal web framework, ~50 lines of code
**Frontend**: Plain HTML/CSS/JavaScript - no complex frameworks
**Communication**: HTTP polling every 2 seconds - no WebSockets complexity
**Storage**: In-memory dictionaries - no database for MVP
**AI Integration**: Direct import of existing agents from agents/ directory

### Why This Stack?

- **Flask**: Already Python, integrates directly with existing game code
- **No Database**: 20 users max, in-memory storage is sufficient  
- **HTTP Polling**: Simple, reliable, no WebSocket complexity for small user base
- **Plain JavaScript**: No build steps, frameworks, or dependencies

## Core Components

### Room Management (Simple)

```python
# Simple in-memory storage
rooms = {}  # room_code: {'players': [], 'game_state': None, 'host': str}
players = {}  # session_id: {'nickname': str, 'room_code': str}

def create_room(nickname):
    room_code = random.randint(1000, 9999)  # 4-digit codes
    session_id = str(uuid.uuid4())[:8]  # Simple session IDs
    rooms[room_code] = {'players': [session_id], 'game_state': None, 'host': session_id}
    players[session_id] = {'nickname': nickname, 'room_code': room_code}
    return room_code, session_id
```

### Game Integration

```python
# Direct integration with existing code
from sim.perudo import PerudoSimulator
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent

# Simple game setup
def start_game(room_code):
    room = rooms[room_code]
    players_list = [players[pid]['nickname'] for pid in room['players']]
    
    # Add one bot for testing
    bot = RandomAgent('Bot')
    game = PerudoSimulator(players_list + ['Bot'])
    room['game_state'] = game
```

## File Structure

```
web/
├── app.py              # Flask app (~100 lines)
├── static/
│   ├── style.css       # Basic styles (~50 lines)
│   └── game.js         # Game interface (~150 lines)
└── templates/
    ├── index.html      # Home page (~30 lines)
    ├── room.html       # Game room (~80 lines)
    └── game.html       # Game interface (~100 lines)
```

**Total estimated code**: ~500 lines (vs 2000+ in original spec)

## API Endpoints (Simple REST)

```
GET  /                  # Home page
POST /create_room       # Create new room  
POST /join_room         # Join existing room
GET  /room/<code>       # Room lobby page
GET  /game/<code>       # Game page
POST /action            # Submit game action
GET  /poll/<code>       # Get current game state
```

## Implementation Plan

### Week 1: Basic Structure
- Create Flask app with room creation/joining
- Simple HTML templates for UI
- Basic room management in memory
- **Tests**: Room creation/joining functionality

### Week 2: Game Integration  
- Integrate with existing PerudoSimulator
- Add one random bot per game
- Implement action submission and game state polling
- **Tests**: Game flow integration, AI agent interaction

### Week 3: UI Polish
- Add basic CSS for readability
- Implement dice display and action buttons
- Add game history and winner display  
- **Tests**: UI functionality, game completion flows

## Simplified Features

### What's Included (MVP)
- Create/join rooms with 4-digit codes
- Play Perudo with 1 bot opponent  
- Basic web interface with dice and actions
- Automatic bot turns
- Win/loss display and restart

### What's Excluded (Future)
- Multiple human players (start with 1v1 vs bot)
- Advanced AI configuration
- User authentication or accounts
- Game history or statistics
- Real-time updates (polling is fine)
- Mobile optimization
- Advanced security features

## Performance Considerations

**Concurrent Games**: 5 max (sufficient for 20 users)
**Memory Usage**: ~10MB for all game states
**Bot Computation**: Use lightweight RandomAgent for MVP
**Polling Frequency**: 2-second intervals (balance between responsiveness and load)

## Testing Strategy (TDD)

### Unit Tests (~20 tests)
- Room creation and joining logic
- Game state management  
- Action validation
- Bot integration

### Integration Tests (~10 tests)
- Full game flow from room creation to completion
- HTTP endpoint functionality
- Game state persistence during polling

### Manual Testing
- Browser compatibility (Chrome/Firefox)
- Basic mobile usability
- Multi-tab simulation of multiple users

## Deployment

### Local Development
```bash
cd web/
python app.py  # Runs on localhost:5000
```

### Simple Sharing (Optional)
- Use ngrok for temporary external access: `ngrok http 5000`
- No complex tunneling or SSL certificates needed for MVP

## Security (Minimal)

**Input Validation**: Basic form validation for nicknames and room codes
**Session Management**: Simple UUID-based sessions (no authentication)
**Rate Limiting**: None needed for 20 users max
**HTTPS**: Not required for local/LAN usage

## Success Criteria

### Functional Requirements Met
- ✅ Create and join game rooms
- ✅ Play complete Perudo game vs AI bot  
- ✅ Web interface shows dice, bids, and actions
- ✅ Game determines winner correctly

### Non-Functional Requirements Met  
- ✅ Supports up to 20 concurrent users
- ✅ No advanced security complexity
- ✅ Under 500 lines of new code
- ✅ Follows TDD with comprehensive tests
- ✅ Uses existing game engine and agents

## Future Expansion Path

When ready to grow beyond MVP:
1. Add multiple human players per room
2. Implement WebSocket for real-time updates  
3. Add Monte Carlo bot with difficulty settings
4. Implement user accounts and game statistics
5. Add mobile-responsive design

This specification prioritizes getting a working multiplayer Perudo game online quickly while respecting the project's educational focus and single-developer constraints.