where python
where pyenv
pyenv version
pyenv versions
pyenv which python

Measure-Command {
    & "$env:USERPROFILE\.pyenv\pyenv-win\shims\python.BAT" -Ic "import platform; print(platform.python_version())"
}