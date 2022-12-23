:- use_module(library(sockets)).

:- dynamic connect_conf/2.
:- assert(connect_conf('127.0.0.1':6379, 'RESPONSE')).

mas_send(X):-
    connect_conf(Host, Channel),
    socket_client_open(Host, S, [type(text)]),
    write(S, 'RPUSH '), write(S, Channel), write(S, ' '),
    write(S, X), nl(S), close(S).

mas_connect_conf(Host, Channel) :-
    retract(connect_conf(_,_)),
    assert(connect_conf(Host, Channel)).