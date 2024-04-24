def render(env, record=False):
    # Slope
    env._viewer.function(0, 1, env._height)

    # Car
    car_body = [
        [-3e-2, 0],
        [-3e-2, 2e-2],
        [-2e-2, 2e-2],
        [-1e-2, 3e-2],
        [1e-2, 3e-2],
        [2e-2, 2e-2],
        [3e-2, 2e-2],
        [3e-2, 0],
    ]

    x_car = (env.state[0] + 1) / 2
    y_car = env._height(x_car)
    c_car = [x_car, y_car]
    angle = env._angle(x_car)
    env._viewer.polygon(c_car, angle, car_body, color=(32, 193, 54))

    frame = env._viewer.get_frame() if record else None

    env._viewer.display(env._dt)

    return frame
