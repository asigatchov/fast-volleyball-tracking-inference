from enum import Enum
from typing import Optional, Tuple

class Team(Enum):
    """Команды"""
    TEAM_A = "Team A"
    TEAM_B = "Team B"

class Action(Enum):
    """Действия во время игры"""
    WIN_POINT = "win_point"      # Выиграно очко
    LOSE_POINT = "lose_point"    # Проиграно очко
    CONTINUE_SERVE = "continue"  # Продолжение подачи
    SWITCH_SERVE = "switch"      # Переход подачи

class GameState(Enum):
    """Состояния игры"""
    SERVING = "serving"          # Подача
    IN_PLAY = "in_play"          # Мяч в игре
    POINT_WON = "point_won"      # Очко выиграно
    SET_OVER = "set_over"        # Партия завершена
    MATCH_OVER = "match_over"    # Матч завершен

class SetResult(Enum):
    """Результат партии"""
    TEAM_A_WON = "team_a_won"
    TEAM_B_WON = "team_b_won"
    IN_PROGRESS = "in_progress"

class VolleyballScoreboard:
    """State-машина для ведения счета в волейболе"""
    
    def __init__(self, points_to_win_set: int = 25, sets_to_win_match: int = 3):
        """
        Инициализация счетчика
        
        Args:
            points_to_win_set: Очков для победы в партии (по умолчанию 25)
            sets_to_win_match: Партий для победы в матче (по умолчанию 3)
        """
        self.points_to_win_set = points_to_win_set
        self.sets_to_win_match = sets_to_win_match
        
        # Текущее состояние игры
        self.state = GameState.SERVING
        
        # Счет текущей партии
        self.current_set_scores = {Team.TEAM_A: 0, Team.TEAM_B: 0}
        
        # Счет по партиям
        self.set_wins = {Team.TEAM_A: 0, Team.TEAM_B: 0}
        
        # История партий
        self.sets_history = []
        
        # Текущая подающая команда
        self.serving_team: Optional[Team] = None
        
        # Номер текущей партии
        self.current_set_number = 1
        
        # Флаг для пятой партии (тай-брейк)
        self.is_tiebreak = False
        
        print(f"Матч начался! Формат: до {sets_to_win_match} побед, в партии до {points_to_win_set} очков")
    
    def start_match(self, first_serve: Team = Team.TEAM_A) -> None:
        """Начало матча"""
        self.serving_team = first_serve
        self.state = GameState.SERVING
        print(f"Матч начался! Первой подает: {self.serving_team.value}")
    
    def start_set(self) -> None:
        """Начало новой партии"""
        self.current_set_scores = {Team.TEAM_A: 0, Team.TEAM_B: 0}
        self.state = GameState.SERVING
        
        # Определяем подающего для новой партии
        if self.current_set_number == 1:
            self.serving_team = Team.TEAM_A
        else:
            # Чередуем первую подачу в партиях
            self.serving_team = (Team.TEAM_B if self.serving_team == Team.TEAM_A 
                               else Team.TEAM_A)
        
        # Проверяем, не пятая ли это партия (тай-брейк)
        if self.set_wins[Team.TEAM_A] == 2 and self.set_wins[Team.TEAM_B] == 2:
            self.is_tiebreak = True
            self.points_to_win_set = 15
            print(f"Тай-брейк! Играем до 15 очков")
        else:
            self.is_tiebreak = False
        
        print(f"Начало {self.current_set_number}-й партии. Подает: {self.serving_team.value}")
    
    def serve(self) -> Action:
        """Обработка подачи"""
        if self.state != GameState.SERVING:
            raise ValueError("Мяч не готов к подаче")
        
        print(f"{self.serving_team.value} подает")
        self.state = GameState.IN_PLAY
        return Action.CONTINUE_SERVE
    
    def process_rally(self, winning_team: Team) -> Action:
        """
        Обработка розыгрыша
        
        Args:
            winning_team: Команда, выигравшая розыгрыш
        
        Returns:
            Action: Результат розыгрыша
        """
        if self.state != GameState.IN_PLAY:
            raise ValueError("Мяч не в игре")
        
        if winning_team == self.serving_team:
            # Подающая команда выиграла розыгрыш
            self.current_set_scores[winning_team] += 1
            print(f"{winning_team.value} выигрывает очко! Счет: {self.get_score()}")
            self.state = GameState.POINT_WON
            
            # Проверяем завершение партии
            if self._is_set_over():
                return self._end_set()
            
            # Подающая команда продолжает подавать
            self.state = GameState.SERVING
            return Action.WIN_POINT
        else:
            # Принимающая команда выиграла розыгрыш
            self.current_set_scores[winning_team] += 1
            print(f"{winning_team.value} выигрывает очко! Счет: {self.get_score()}")
            self.state = GameState.POINT_WON
            
            # Переход подачи
            self.serving_team = winning_team
            
            # Проверяем завершение партии
            if self._is_set_over():
                return self._end_set()
            
            self.state = GameState.SERVING
            return Action.SWITCH_SERVE
    
    def _is_set_over(self) -> bool:
        """Проверка завершения партии"""
        score_a = self.current_set_scores[Team.TEAM_A]
        score_b = self.current_set_scores[Team.TEAM_B]
        
        if self.is_tiebreak:
            # Тай-брейк: минимум 15 очков и разница минимум 2
            return (score_a >= 15 or score_b >= 15) and abs(score_a - score_b) >= 2
        else:
            # Обычная партия: минимум 25 очков и разница минимум 2
            return (score_a >= self.points_to_win_set or score_b >= self.points_to_win_set) and abs(score_a - score_b) >= 2
    
    def _end_set(self) -> Action:
        """Завершение партии"""
        self.state = GameState.SET_OVER
        
        # Определяем победителя партии
        if self.current_set_scores[Team.TEAM_A] > self.current_set_scores[Team.TEAM_B]:
            winner = Team.TEAM_A
        else:
            winner = Team.TEAM_B
        
        # Сохраняем результат партии
        self.set_wins[winner] += 1
        set_result = {
            "set_number": self.current_set_number,
            "score": self.current_set_scores.copy(),
            "winner": winner
        }
        self.sets_history.append(set_result)
        
        print(f"Партия {self.current_set_number} завершена!")
        print(f"Победитель: {winner.value}")
        print(f"Счет матча: {self.set_wins[Team.TEAM_A]} - {self.set_wins[Team.TEAM_B]}")
        
        # Проверяем завершение матча
        if self._is_match_over():
            return self._end_match()
        
        # Начинаем следующую партию
        self.current_set_number += 1
        return Action.WIN_POINT
    
    def _is_match_over(self) -> bool:
        """Проверка завершения матча"""
        return (self.set_wins[Team.TEAM_A] >= self.sets_to_win_match or 
                self.set_wins[Team.TEAM_B] >= self.sets_to_win_match)
    
    def _end_match(self) -> Action:
        """Завершение матча"""
        self.state = GameState.MATCH_OVER
        
        if self.set_wins[Team.TEAM_A] > self.set_wins[Team.TEAM_B]:
            winner = Team.TEAM_A
        else:
            winner = Team.TEAM_B
        
        print(f"=" * 50)
        print(f"МАТЧ ЗАВЕРШЕН!")
        print(f"ПОБЕДИТЕЛЬ: {winner.value}")
        print(f"Итоговый счет: {self.set_wins[Team.TEAM_A]} - {self.set_wins[Team.TEAM_B]}")
        print(f"=" * 50)
        
        return Action.WIN_POINT
    
    def get_score(self) -> str:
        """Получить текущий счет"""
        return f"{self.current_set_scores[Team.TEAM_A]}-{self.current_set_scores[Team.TEAM_B]}"
    
    def get_match_score(self) -> str:
        """Получить счет матча по партиям"""
        return f"{self.set_wins[Team.TEAM_A]} - {self.set_wins[Team.TEAM_B]}"
    
    def get_state(self) -> GameState:
        """Получить текущее состояние игры"""
        return self.state
    
    def get_serving_team(self) -> Optional[Team]:
        """Получить текущую подающую команду"""
        return self.serving_team
    
    def get_current_set_number(self) -> int:
        """Получить номер текущей партии"""
        return self.current_set_number
    
    def get_sets_history(self) -> list:
        """Получить историю партий"""
        return self.sets_history.copy()
    
    def is_match_over(self) -> bool:
        """Проверить, завершен ли матч"""
        return self.state == GameState.MATCH_OVER
    
    def reset(self) -> None:
        """Сброс счетчика к начальному состоянию"""
        self.__init__(self.points_to_win_set, self.sets_to_win_match)


# Пример использования
def example_usage():
    """Пример использования state-машины"""
    
    # Создаем счетчик для матча до 2 побед, в партии до 25 очков
    scoreboard = VolleyballScoreboard(points_to_win_set=25, sets_to_win_match=2)
    
    # Начинаем матч
    scoreboard.start_match(first_serve=Team.TEAM_A)
    
    # Имитация нескольких розыгрышей
    actions = [
        (Team.TEAM_A, "Team A выигрывает свой розыгрыш подачи"),
        (Team.TEAM_A, "Team A снова выигрывает"),
        (Team.TEAM_B, "Team B выигрывает розыгрыш и перехватывает подачу"),
        (Team.TEAM_B, "Team B выигрывает свою подачу"),
        (Team.TEAM_A, "Team A выигрывает и возвращает подачу"),
    ]
    
    for team, description in actions:
        print(f"\n{description}")
        
        if scoreboard.get_state() == GameState.SERVING:
            scoreboard.serve()
        
        action = scoreboard.process_rally(team)
        print(f"Действие: {action.value}")
        
        if scoreboard.is_match_over():
            break
    
    # Показать текущий статус
    print(f"\nТекущий статус:")
    print(f"Партия: {scoreboard.get_current_set_number()}")
    print(f"Счет в партии: {scoreboard.get_score()}")
    print(f"Счет матча: {scoreboard.get_match_score()}")
    print(f"Подает: {scoreboard.get_serving_team().value if scoreboard.get_serving_team() else 'Не определено'}")
    print(f"Состояние: {scoreboard.get_state().value}")


if __name__ == "__main__":
    example_usage()
