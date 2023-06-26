#!/usr/bin/env python


def get_args(self, exclude=["self", "__class__", "kwargs"], **kwargs):
    args = {key: kwargs[key] for key in kwargs if not key in exclude}
    if "kwargs" in kwargs:
        for key, val in kwargs["kwargs"].items():
            args[key] = val
    return args


def parse_params(user_params, default_params):
    if not user_params:
        params = default_params

    elif type(user_params) == dict:
        params = {}
        for key, val in default_params.items():
            try:
                params[key] = user_params[key]
            except:
                params[key] = val
    else:
        print(
            "@@\tUser hyperparams not understood, resorting to default... [WARNING] "
        )
        params = default_params
    return params
