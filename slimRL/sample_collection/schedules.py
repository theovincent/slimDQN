def linear_schedule(end_e: float, duration: int, t: int, start_e: float = 1.0):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
