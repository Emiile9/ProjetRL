import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle


def load_stats(file_path):
    with open(file_path, "rb") as f:
        stats = pickle.load(f)
    return stats


def plot_stats(stats, title):
    df = pd.DataFrame(stats)
    df["average_over_100"] = df["reward"].rolling(window=100).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df["episode"], y=df["reward"], name="Total Reward", yaxis="y1")
    )

    fig.add_trace(
        go.Scatter(
            x=df["episode"],
            y=df["average_over_100"],
            name="Average Reward (100 eps)",
            yaxis="y1",
        )
    )

    #fig.add_trace(
    #    go.Scatter(x=df["episode"], y=df["epsilon"], name="Epsilon", yaxis="y2")
    #)

    fig.update_layout(
        title=title,
        xaxis=dict(title="Episode"),
        yaxis=dict(title="Reward", rangemode="tozero"),
        yaxis2=dict(title="Epsilon", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(x=0.01, y=0.99),
    )

    return fig

def plot_comparaison(files_list):
    fig = go.Figure()
    # Palette de couleurs qualitative pour distinguer les expériences
    colors = px.colors.qualitative.Plotly

    for i, file_path in enumerate(files_list):
        color = colors[i % len(colors)]
        
        # Chargement des données (remplace par ta fonction load_stats)
        stats = load_stats(file_path)
        df = pd.DataFrame(stats)

        # Calcul de la moyenne mobile
        df["average_over_100"] = df["reward"].rolling(
            window=100,
            min_periods=1
        ).mean()

        # Nettoyage du nom du fichier pour la légende
        label = file_path.replace("stats_epsdiv1000_", "").replace(".pkl", "")

        # --- TRACE 1 : REWARD (Axe Y principal) ---
        fig.add_trace(
            go.Scatter(
                x=df["episode"],
                y=df["average_over_100"],
                mode="lines",
                name=f"Reward {label}",
                line=dict(color=color, width=2),
                legendgroup=label,  # Groupe les deux traces ensemble
            )
        )

        # --- TRACE 2 : EPSILON (Axe Y secondaire) ---
        fig.add_trace(
            go.Scatter(
                x=df["episode"],
                y=df["epsilon"],
                mode="lines",
                name=f"Epsilon {label}",
                yaxis="y2",         # Assignation à l'axe de droite
                line=dict(color=color, dash="dot", width=1),
                opacity=0.6,        # Plus discret pour ne pas surcharger
                legendgroup=label,  # Lié au reward dans la légende
                showlegend=True     # Garder à True car chaque epsilon est unique ici
            )
        )

    # Configuration des axes et du design
    fig.update_layout(
        title="Comparaison des performances et des stratégies d'exploration",
        xaxis=dict(title="Épisode", gridcolor='lightgray'),
        
        # Axe Y principal (Gauche) : Rewards
        yaxis=dict(
            title="Reward moyen (100 eps)",
            rangemode="tozero",
            side="left"
        ),
        
        # Axe Y secondaire (Droite) : Epsilon
        yaxis2=dict(
            title="Valeur d'Epsilon",
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, 1.05], # Epsilon est compris entre 0 et 1
            showgrid=False   # On cache la grille pour éviter la confusion avec l'axe Y1
        ),
        
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified" # Permet de voir toutes les valeurs au même épisode au survol
    )

    return fig
stats_lr_0_001 = load_stats("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo_lr_0_001.pkl")
stats_lr_0_0001 = load_stats("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo_lr_0_0001.pkl")
stats_lr_5e_05 = load_stats("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo_lr_5e-05.pkl")
stats_dict = {"episode": list(range(len(stats_lr_0_001)))[:4000], "reward lr 0.001": stats_lr_0_001[:4000], "reward lr 0.0001": stats_lr_0_0001[:4000], "reward lr 5e-05": stats_lr_5e_05[:4000]}
for key in stats_dict.keys():
    print(f"{key}: {len(stats_dict[key])} entries")

df = pd.DataFrame(stats_dict)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["episode"], y=df["reward lr 0.001"].rolling(
            window=100,
            min_periods=1
        ).mean(), mode="lines", name="Reward lr 0.001"))
fig.add_trace(go.Scatter(x=df["episode"], y=df["reward lr 0.0001"].rolling(
            window=100,
            min_periods=1
        ).mean(), mode="lines", name="Reward lr 0.0001"))
fig.add_trace(go.Scatter(x=df["episode"], y=df["reward lr 5e-05"].rolling(
            window=100,
            min_periods=1
        ).mean(), mode="lines", name="Reward lr 5e-05"))
fig.update_layout(title="Comparaison des rewards pour différents learning rates", xaxis_title="Episode", yaxis_title="Reward moyen (100 eps)", template="plotly_white")
fig.show()

stats_entropy_0_01 = load_stats("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo_entropy_coef_0_01.pkl")
stats_entropy_0_001 = load_stats("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo_entropy_coef_0_001.pkl")
stats_entropy_0_05 = load_stats("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo_entropy_coef_0_05.pkl")
stats_dict = {"episode": list(range(len(stats_entropy_0_01)))[:4000], "reward entropy 0.01": stats_entropy_0_01[:4000], "reward entropy 0.001": stats_entropy_0_001[:4000], "reward entropy 0.05": stats_entropy_0_05[:4000]}
for key in stats_dict.keys():
    print(f"{key}: {len(stats_dict[key])} entries")

df = pd.DataFrame(stats_dict)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["episode"], y=df["reward entropy 0.01"].rolling(
            window=100,
            min_periods=1
        ).mean(), mode="lines", name="Reward entropy 0.01"))
fig.add_trace(go.Scatter(x=df["episode"], y=df["reward entropy 0.001"].rolling(
            window=100,
            min_periods=1
        ).mean(), mode="lines", name="Reward entropy 0.001"))
fig.add_trace(go.Scatter(x=df["episode"], y=df["reward entropy 0.05"].rolling(
            window=100,
            min_periods=1
        ).mean(), mode="lines", name="Reward entropy 0.05"))
fig.update_layout(title="Comparaison des rewards pour différentes entropies", xaxis_title="Episode", yaxis_title="Reward moyen (100 eps)", template="plotly_white")
fig.show()

stats_ppo = load_stats("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo.pkl")
dict_stats_ppo = {"episode": list(range(len(stats_ppo))), "reward": stats_ppo}
df = pd.DataFrame(dict_stats_ppo) 
plot_stats(df, "PPO - Reward par épisode").show()

stats_dqn = load_stats("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats_dqn_final.pkl")
dict_stats_dqn = {"episode": list(range(len(stats_dqn["episode"]))), "reward": stats_dqn["reward"], "epsilon": stats_dqn["epsilon"]} 
df_dqn = pd.DataFrame(dict_stats_dqn)

fig = go.Figure() 
fig.add_trace(go.Scatter(x=df_dqn["episode"], y=df_dqn["reward"].rolling( window=100, min_periods=1 ).mean(), mode="lines", name="Reward DQN (moyenne 100 eps)")) 
fig.add_trace(go.Scatter(x=df["episode"], y=df["reward"].rolling( window=100, min_periods=1 ).mean(), mode="lines", name="Reward PPO (moyenne 100 eps)")) 
fig.update_layout(title="Comparaison des rewards DQN et PPO", xaxis_title="Episode", yaxis_title="Reward", template="plotly_white") 
fig.show()