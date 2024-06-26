- `commit` - фиксирует изменения в репозитории в виде коммита
- `branch 
	- `[name]` -  создать ветку
	- `-f [branch_from] [branch_or_commit_to]` - **branch forcing** - перемещение ветки `[branch_from]` на то место, которое указывается в `[branch_or_commit_to]`
	  Пример: `branch -f main HEAD~3`
- `checkout [name]` - переключиться на ветку
	- `-b `- создать, если не создана
	- `^` - переключиться на родительский коммит (В качестве `[name]` можно указать **HEAD**. Тогда движение будет производиться относительно текущей ссылки *HEAD*) (
		- `checkout [hash]` - переключение на ссылку на коммит (**HEAD**). Весь хеш можно не записывать, а только первые 4 символа
	- `^^` - переключиться на коммит прародителя
	- `~<num>` - переключиться на num коммитов ранее
- `merge [имя ветки, откуда сливать изменения]` - провести слияние указанной ветки с той, на которой вызывается команда. В результате получается коммит-merge, которые имеет два родителя.
	- Чтобы сравнять версию ветки [name] с [main], можно сделать трюк: `git checkout [name]; git merge main`
- `rebase [main]` - копирование коммитов из текущей ветки в ветку [main]
	- `git checkout [main]; git rebase [name]` - обновление ветки [main] до новых обновлений после *rebase*
- `reset [ссылка_куда_переместиться]` - отменяет изменения, перенося ссылку на ветку *(на которой в данный момент работаем)* назад, переписывая историю (как будто некоторых коммитов вовсе и не было)
  **Работает только на локальных ветках, в локальных репозиториях!**
- `revert [ссылка_что_отменить]` - делает новый коммит, полностью противоположный тем изменениям, которые мы хотим отменить

# Подключение к удаленному репозиторию по ssh
[Источник](https://habr.com/ru/articles/755036/)

1. Генерация ssh ключей или их проверка
	1. Местоположение: ~/.ssh/id_method и id_method.pub (id_ed25519, id_rsa или другой метод генерации ассиметричных ключей)
	2. Генерация с помощью `openssh`: `ssh-keygen -t ed25519` (`ed25519` - метод)
	3. Настройка конфига:  github.com - url или ip сервиса, к которому будем подключаться
	```
	Host github.com    
		HostName github.com    
		User git    
		IdentityFile ~/.ssh/id_method    
		IdentitiesOnly yes
	```
2. [Добавление публичного ключа на сервис](https://github.com/settings/keys)
3. Добавить сервис в список доверенных хостов: `ssh -T git@github.com`
4. Клонировать репозиторий по ssh ссылке или
	1. Поменять адрес удалённого репозитория: `git remote set-url origin git@serviceurl:username/reponame.git`
	   serviceurl - github.com
	   username - ник владельца репозитория (Luzinsan)
	   reponame - ну тут понятно
	2. Проверка адресов удалённых репозиториев: `git remote -v` 
5. Ок:
```
origin	git@github.com:Luzinsan/petrol.git (fetch)
origin	git@github.com:Luzinsan/petrol.git (push)
```

## Это надо знать
- git add . - прикрепить в коммит
- git commit -m 'latinica tuta doljna bit'
- git remote - показать удалённые репы
- git push origin main - origin(удалённая репа), main(ветка в этой репе)